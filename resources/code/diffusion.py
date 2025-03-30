import numpy as np
import torch
import torch.nn as nn
import wandb
import os
import argparse
import hashlib
import math

from diffusers import DDPMScheduler

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """
    An abstract base class for all loggers, incluing python's logger,
    wandb, Tensorboard, etc.\n If you want to create a custom logger, you should inherit this class, and implement all abstract method indicated by @classmethod.
    """
    @abstractmethod
    def log(self, *args, **kwargs):
        pass
    
    @classmethod
    def create(logger_cls, *args, **kwargs):
        if not issubclass(logger_cls, BaseLogger):
            raise TypeError(f'Class [{logger_cls.__name__}] is not derived from [BaseLogger].')
        return logger_cls(*args, **kwargs)

class WandbLogger(BaseLogger):
    def __init__(self, key, entity, name, project, config = None):
        self.key     = key
        self.entity  = entity
        self.name    = name
        self.project = project
        self.id      = self.__generate_id_from_name(self.name)
        self.config  = self.__args_to_dict(config) if config else None
    
    def login(self):
        wandb.login(key = self.key)
    
    def init(self, rewind = False):
        """
        Initialize a new run depending on parameter `rewind`. If `rewind=True`, will set `resume=None` which overwrites the content of the run specified by `id`.
        This can be used when you delete local experiment files and want to use the same experiment name without changing the wandb id.
        If `rewind=False`, `resume` will be set to `allow` which will resume from the last step of the run specified by `id`.
        """
        wandb.init(
            entity  = self.entity,
            project = self.project,
            name    = self.name,
            config  = self.config,
            id      = self.id,
            resume  = "allow" if not rewind else None
        )

    def finish():
        wandb.finish()

    def log(self, log, step = None):
        """
        Log a dict.
        """
        assert isinstance(log, dict), f'Input log has type {type(log)} rather than dict.'
        wandb.log({k: v for k, v in log.items()}, step = step)

    def log_image_tensor(self, key, image, step = None, **others):
        """
        Log an array of images of torch.Tensor, should have size (N, C, W, H).
        """
        image = self.__array2grid(image)
        wandb.log({key: wandb.Image(image), **others}, step = step)

    def delete_run(self):
        """
        Delete current run.
        """
        api = wandb.Api()
        try:
            run = api.run(f'{self.entity}/{self.project}/{self.id}')
            run.delete()
        except Exception as e:
            print('Wandb:', e)            

    def __args_to_dict(self, args):
        return {
            k: self.__args_to_dict(v) if isinstance(v, argparse.Namespace) else v
            for k, v in vars(args).items()
        }

    def __generate_id_from_name(self, name):
        return str(int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

    def __array2grid(self, x):
        n_row = round(math.sqrt(x.size(0)))
        x = make_grid(x, nrow = n_row, normalize = True, value_range = (-1, 1))
        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return x

########################################################################
###                       Checkpoint Saver                           ###
###  A checkpoint saver does the following things every N steps:     ###
###  1). Log everyhing by your logger, e.g., WanDB, Tensorboard,     ###
###  2). Save the model and other information into local .pt file,   ###
###    2.1) Current training steps `steps`,                          ###
###    2.2) Model status dict `model`,                               ###
###    2.3) Optimizer status `optimizer`,                            ###
###    2.4) Args for this run `args`,                                ###
###    2.5) Other necessary components such as ema `ema`.            ###
########################################################################

class CheckpointSaver():
    def __init__(self, current_step, save_every, autosave_path):
        """
        Initialize a checkpoint saver by ccurrent_step, save_every and autosave_path.
        autosave_path is the default save path if no path is provided when calling save().
        """
        assert current_step % save_every == 0, f'{current_step} % {save_every} != 0, current_step should be a multiply of save_every'

        self.steps          = current_step
        self.save_every     = save_every
        self.autosave_path  = autosave_path

    def save(self, ckpt, auto_save = False, save_path = None):
        assert isinstance(ckpt, dict), 'The input ckpt must be a dict.'
        assert 'model' in ckpt.keys(), 'Key `model` must be included in your provided ckpt.'
        assert 'steps' in ckpt.keys(), 'Key `steps` must be included in your provided ckpt.'
        assert 'optimizer' in ckpt.keys(), 'Key `optimizer` must be included in your provided ckpt.'
        assert ckpt['steps'] % self.save_every == 0, f'Key `steps` should be a multiplier of `save_every`. {ckpt["steps"]} % {self.save_every} != 0.'
        assert auto_save == True or (auto_save == False and save_path is not None), 'If save_auto is False, you must provide a valid save_path.'

        if auto_save == True:
            save_path = self.autosave_path
    
        torch.save(ckpt, save_path)


########################################################################
###                       Checkpoint Loader                          ###
###  A checkpoint loader loads checkpoint from a given path.         ###
###  You should specify what components (by string) to retrieve from ###
###  the checkpoint file. Should contain at least keys `steps`,      ###
###  `model` and `optimizer`.                                        ###
########################################################################

class CheckpointLoader():
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def load(self, *args):
        """
        Load checkpoint file and retrieve components (model, steps, etc) specified by the input args.
        """
        if not os.path.exists(self.ckpt_path):
            return None

        assert 'steps' in args, 'Key `steps` must be included in your provided args.'
        assert 'model' in args, 'Key `model` must be included in your provided args.'
        assert 'optimizer' in args, 'Key `optimizer` must be included in your provided args.'

        ckpt = torch.load(self.ckpt_path, map_location = lambda storage, loc: storage)
        return {arg: ckpt[arg] for arg in args}


########################################################################
###                       Model Definitions                          ###
###  Includes:                                                       ###
###  1). Timestep Embedder.                                          ###
###  2). Resnet Block.                                               ###
###  3). Attention Block.                                            ###
###  4). Downsample and Upsample.                                    ###
###  5). UNet.                                                       ###
########################################################################

def Normalize(in_channels, num_groups = 32):
    return torch.nn.GroupNorm(num_groups = num_groups, num_channels = in_channels, eps = 1e-6, affine = True)

class TimestepEmbedder(nn.Module):
    """
    > Embeds scalar timesteps into vector representations.\n
    > timestep -> frequency_embedding_size -> hidden_size.
    """
    def __init__(self, hidden_size, frequency_embedding_size = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias = True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, frequency_embedding_size, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param frequency_embedding_size: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start = 0, end = half, dtype = torch.float32) / half
        ).to(device = t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim = -1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResnetBlock(nn.Module):
    def __init__(self, *, 
                 in_channels, 
                 out_channels = None, 
                 time_channels = 128,
                 conv_shortcut = False
        ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        self.silu1 = nn.SiLU()

        self.time_proj = nn.Linear(time_channels, out_channels)
        self.time_silu = nn.SiLU()
        
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1) 
        self.silu2 = nn.SiLU()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size = 1,
                                              stride = 1,
                                              padding = 0) 
                
    def forward(self, x, t):
        """
        x is of shape (B, C, H, W), t is of shape (B, T)
        """
        h = x
        h = self.norm1(h)
        h = self.silu1(h)
        h = self.conv1(h)

        if t is not None:
            h = h + self.time_proj(self.time_silu(t))[:, :, None, None]

        h = self.norm2(h)
        h = self.silu2(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x + h

class AttnBlock(nn.Module):
    def __init__(
            self,
            in_channels: int
        ):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0) 
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0) 
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0) 
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        
    def forward(self, x):
        h = x
        h = self.norm(h)
        q, k, v = self.q(h), self.k(h), self.v(h)

        B, C, H, W = q.shape
        q = q.reshape(B, C, -1).permute(0, 2, 1) # [B, HW, C]
        k = k.reshape(B, C, -1)                  # [B, C, HW]
        a = torch.bmm(q, k)
        a = a * (int(C) ** (-0.5))
        a = nn.functional.softmax(a, dim = 2)

        v = v.reshape(B, C, -1)                  # [B, C, HW]
        a = a.permute(0, 2, 1)
        h = torch.bmm(v, a)
        h = h.reshape(B, C, H, W)

        h = self.proj_out(h)

        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv = True):
        super().__init__()

        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size = 3,
                                  stride = 2,
                                  padding = 0)
    
    def forward(self, x):
        if self.with_conv:
            padding = (0, 1, 0, 1)
            x = nn.functional.pad(x, padding, mode = 'constant', value = 0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size = 2, stride = 2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv = True):
        super().__init__()

        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = 1)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor = 2.0, mode = 'nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            in_size: int,   
            in_channels: int = 256,
            latent_size: int = 3,
            channel_multiplier: tuple = (1, 2, 4),
            num_residual_blks: int = 2
        ):
        super().__init__()

        # Input image size, generally 64
        self.in_size = in_size
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.num_resolutions = len(channel_multiplier)
        self.num_residule_blks = num_residual_blks

        self.time_channels = in_channels
        self.time_embedder = TimestepEmbedder(self.time_channels)

        self.out_channels = latent_size
        self.in_conv = \
                nn.Conv2d(latent_size, in_channels, kernel_size = 3, stride = 1, padding = 1) 
        
        # Downsampling
        current_size = in_size
        in_channel_multiplier = (1, ) + tuple(channel_multiplier)   

        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = in_channels * in_channel_multiplier[level]
            block_out = in_channels * channel_multiplier[level]
            for _ in range(num_residual_blks):
                block.append(ResnetBlock(
                    in_channels = block_in,
                    out_channels = block_out,
                    time_channels = self.time_channels
                ))
                block_in = block_out
                if current_size <= 16:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                current_size = current_size // 2
            
            self.down.append(down)
        
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels = block_in,
            out_channels = block_in,
            time_channels = self.time_channels
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels = block_in,
            out_channels = block_in,
            time_channels = self.time_channels
        )

        # Upsampling
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = in_channels * channel_multiplier[level]
            skip_in = in_channels * channel_multiplier[level]
            for _ in range(num_residual_blks + 1):
                if _ == self.num_residule_blks:
                    skip_in = in_channels * in_channel_multiplier[level]
                block.append(ResnetBlock(
                    in_channels = block_in + skip_in,
                    out_channels = block_out,
                    time_channels = self.time_channels
                ))
                block_in = block_out
                if current_size <= 16:
                    attn.append(AttnBlock(block_in))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(block_in)
                current_size = current_size * 2
            self.up.insert(0, up)
    
        # End
        self.norm_out = Normalize(block_in)
        self.silu_out = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_in,
            self.out_channels,
            kernel_size = 3,
            stride = 1, 
            padding = 1
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std = 0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std = 0.02)
    
    def forward(self, x, t):
        """
        x is of shape (B, latent_size, H, W).
        t is of shape (B, ).
        """
        t = self.time_embedder(t)

        # Downsampling
        hs = [self.in_conv(x)]
        for level in range(self.num_resolutions):
            for block in range(self.num_residule_blks):
                h = self.down[level].block[block](hs[-1], t)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[block](h)
                hs.append(h)
            if level != self.num_resolutions - 1:
                hs.append(self.down[level].downsample(hs[-1]))
        
        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, t)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, t)

        # Upsampling
        for level in reversed(range(self.num_resolutions)):
            for block in range(self.num_residule_blks + 1):
                h = self.up[level].block[block](torch.cat([h, hs.pop()], dim = 1), t)
                if len(self.up[level].attn) > 0:
                    h = self.up[level].attn[block](h)
            if level != 0:
                h = self.up[level].upsample(h)
        
        # End
        h = self.norm_out(h)
        h = self.silu_out(h)
        h = self.conv_out(h)

        return h 

def main(args):
    data_path     = args.data_path
    results_dir   = args.results_dir
    exp_name      = args.exp_name
    autosave_name = args.autosave_name

    image_size    = args.image_size
    epochs        = args.epochs
    batch_size    = args.batch_size
    save_every    = args.save_every

    wandb         = args.wandb
    wandb_key     = args.wandb_key
    wandb_entity  = args.wandb_entity
    wandb_project = args.wandb_project

    device = "cuda"

    # Create experiment directory
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok = True)

    # Create model, optimizer and track steps
    model = UNet(image_size)
    steps = 0

    # Resume checkpoint if any
    ckpt_autosave_path = os.path.join(exp_dir, f'{autosave_name}.pt')
    ckpt_loader = CheckpointLoader(ckpt_autosave_path)
    ckpt = ckpt_loader.load('model', 'steps', 'optimizer')
    if ckpt:
        model.load_state_dict(ckpt['model'])
        steps = ckpt['steps']

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Create and enable wandb logger if specified
    if wandb:
        wandb_logger = WandbLogger(wandb_key, wandb_entity, exp_name, wandb_project, args)
        wandb_logger.login()
        rewind = True if ckpt is None else False
        wandb_logger.init(rewind)

    # Define image transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], inplace = True)
    ])

    # Define dataset and dataloader
    dataset = ImageFolder(data_path, transform = transform)
    dataloder = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, drop_last = True)

    # Define DDPM scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps = 1000, 
        beta_schedule = 'scaled_linear', 
        clip_sample = False,
        steps_offset = 1
    )

    # Create a checkpoint saver to save checkpoint and logs
    ckpt_saver = CheckpointSaver(steps, save_every, ckpt_autosave_path)

    # Track loss
    running_loss = 0.

    ### Begin training
    model.train()
    for _ in range(epochs):
        for x, _ in dataloder:
            x = x.to(device)
            t = torch.randint(0, scheduler.num_train_timesteps, (x.shape[0], ), device = device)
            noise = torch.randn_like(x).to(device)
            x_noised = scheduler.add_noise(x, noise, t)
            pred_noise = model(x_noised, t)

            loss = nn.functional.mse_loss(pred_noise, noise)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            print(f'Running step {steps}')
            if steps % save_every == 0:
                # Logging
                avg_loss = running_loss / save_every
                log = {
                    'avg_loss': avg_loss,
                }
                if wandb:
                    wandb_logger.log(log, steps)
                else:
                    print(f'avg_loss: {avg_loss}')

                running_loss = 0.

                # Save checkpoint
                ckpt = {
                    'model': model.state_dict(),
                    'steps': steps,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }
                ckpt_saver.save(ckpt, True)

                # Sample
                model.eval()
                with torch.no_grad():
                    scheduler.set_timesteps(50, device = device)
                    denoise = torch.randn_like(x).to(device)
                    for t in scheduler.timesteps:
                        timestep = torch.tensor(t, device = device).long().repeat(denoise.shape[0])
                        predicted_noise = model(denoise, timestep)
                        denoise = scheduler.step(predicted_noise, t, denoise, return_dict = False)[0]

                    samples = denoise
                    if wandb:
                        wandb_logger.log_image_tensor('sample', samples, steps = steps)
                    else:
                        save_image(samples, os.path.join(exp_dir, f'{steps}.png', nrow = math.sqrt(samples.shape[0]), normalize = True, value_range = (-1, 1)))
                model.train()
    
    if wandb:
        wandb_logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, default = './data/anime', help = 'Directory to the dataset, for example `./data`.')
    parser.add_argument('--results-dir', type = str, default = './results', help = 'Directory to store logs, checkpoints and results.')
    parser.add_argument('--exp-name', type = str, required = True, help = 'Experiment name, should be unique for each different runs. Use the same name if you want to resume previous runs.')
    parser.add_argument('--autosave-name', type = str, default = 'ckpt', help = 'The file name that will be used for autosave and resume. You do not need to change this.')

    parser.add_argument('--image-size',  type = int, choices = [32, 64], default = 64)
    parser.add_argument('--epochs', type = int, default = 400)
    parser.add_argument('--batch-size', type = int, default = 8)
    parser.add_argument('--save-every', type = int, default = 200)
    
    parser.add_argument('--wandb', action = 'store_true', help = 'Whether to use wandb for logging. If enabled, you should provide key, entity and project arguments.')
    parser.add_argument('--wandb_key', type = str, default = 'YOUR_KEY_HERE', help = "Wandb key.")
    parser.add_argument('--wandb_entity', type = str, default = 'YOUR_ENTITY_HERE', help = 'Wandb entity.')
    parser.add_argument('--wandb_project', type = str, default = 'YOUR_PROJECT_HERE', help = 'Wandb project.')

    args = parser.parse_args()
    main(args)