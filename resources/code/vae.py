import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import wandb
import hashlib
import math

from einops.layers.torch import Rearrange
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.distributions import Normal
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
    
    def init(self):
        wandb.init(
            entity  = self.entity,
            project = self.project,
            name    = self.name,
            config  = self.config,
            id      = self.id,
            resume  = "allow"
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
    
class DownSample2D(nn.Module):
    """
    Upsample a tensor of size (B, C_in, H, W) to (B, C_out, H / 2, W / 2).
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.downsample = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1 = 2, p2 = 2),
            nn.Conv2d(dim_in * 4, dim_out, 1)
        )

    def forward(self, x):
        return self.downsample(x)

class UpSample2D(nn.Module):
    """
    Upsample a tensor of size (B, C_in, H, W) to (B, C_out, H * 2, W * 2).
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        )

    def forward(self, x):
        return self.upsample(x)

class VAE(nn.Module):
    def __init__(self, image_zize, n_channel, hidden_dim, device):
        super().__init__()
        self.image_size = image_zize
        self.distribution = Normal(0, 1)
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channel, hidden_dim // 8, 3, 1, 1),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim // 8),
            DownSample2D(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim // 4),
            DownSample2D(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim // 2),
            DownSample2D(hidden_dim // 2, hidden_dim),
            nn.MaxPool2d(image_zize // 8),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            Rearrange("b (c x y) -> b c x y", x = 2, y = 2),
            nn.BatchNorm2d(hidden_dim),
            UpSample2D(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim),
            UpSample2D(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim),
            UpSample2D(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim // 2),
            UpSample2D(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.BatchNorm2d(hidden_dim // 4),
            UpSample2D(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 8, n_channel, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        mean, log_std = torch.chunk(x, chunks = 2, dim = 1)
        std = torch.exp(log_std)
        z = mean + std * self.distribution.sample(mean.shape).to(self.device)
        x = self.decoder(z)
        return mean, std, x
    
    def sample(self, size):
        return self.decoder(self.distribution.sample(size).to(self.device))
    
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mean, std):
        return 0.5 * (std ** 2 + mean ** 2 - torch.log(std ** 2) - 1).sum(axis = 1).mean()
    
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
    model = VAE(image_size, 3, 512, device)
    steps = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)

    # Resume checkpoint if any
    ckpt_autosave_path = os.path.join(exp_dir, f'{autosave_name}.pt')
    ckpt_loader = CheckpointLoader(ckpt_autosave_path)
    ckpt = ckpt_loader.load(['model', 'steps', 'optimizer'])
    if ckpt is None:
        model.to(device)
    else:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        steps = ckpt['steps']

    # Create and enable wandb logger if specified
    if wandb:
        wandb_logger = WandbLogger(wandb_key, wandb_entity, exp_name, wandb_project, args)
        wandb_logger.login()
        if ckpt is None:
            wandb_logger.delete_run()
        wandb_logger.init()

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

    # Define loss functions
    loss_fn_x = nn.MSELoss()
    loss_fn_z = KLLoss()
    running_loss_x = 0.
    running_loss_z = 0.

    # Create a checkpoint saver to save checkpoint and logs
    ckpt_saver = CheckpointSaver(steps, save_every, ckpt_autosave_path)

    ### Begin training
    model.train()
    for _ in range(epochs):
        for x, _ in dataloder:
            x = x.to(device)
            mean, std, y = model(x)

            loss_x = loss_fn_x(x, y)
            loss_z = loss_fn_z(mean, std)
            running_loss_x += loss_x.item()
            running_loss_z += loss_z.item()

            loss = loss_x + 0.001 * loss_z
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            print(f'Running step {steps}')
            if steps % save_every == 0:
                # Logging
                avg_loss_x = running_loss_x / save_every
                avg_loss_z = running_loss_z / save_every
                log = {
                    'avg_loss_x': avg_loss_x,
                    'avg_loss_z': avg_loss_z
                }
                if wandb:
                    wandb_logger.log(log, steps)
                else:
                    print(f'avg_loss_x: {avg_loss_x}, avg_loss_z: {avg_loss_z}')

                running_loss_x = 0.
                running_loss_z = 0.

                # Save checkpoint
                ckpt = {
                    'model': model.state_dict(),
                    'steps': steps,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }
                ckpt_saver.save(ckpt, True)
                ckpt_saver.save(ckpt, False, os.path.join(exp_dir, f'{steps}.pt'))

                # Sample
                model.eval()
                with torch.no_grad():
                    samples = model.sample((25, 512))
                    samples.cpu().numpy()
                    if wandb:
                        wandb_logger.log_image_tensor('sample', samples, steps = steps)
                    else:
                        save_image(samples, os.path.join(exp_dir, f'{steps}.png', nrow = math.sqrt(samples.shape[0]), normalize = True, value_range = (-1, 1)))
                model.train()
    
    if wandb:
        wandb_logger.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, required = True, help = 'Directory to the dataset, for example `./data`.')
    parser.add_argument('--results-dir', type = str, default = './results', help = 'Directory to store logs, checkpoints and results.')
    parser.add_argument('--exp-name', type = str, required = True, help = 'Experiment name, should be unique for each different runs. Use the same name if you want to resume previous runs.')
    parser.add_argument('--autosave-name', type = str, default = 'ckpt', help = 'The file name that will be used for autosave and resume. You do not need to change this.')

    parser.add_argument('--image-size',  type = int, choices = [32, 64], default = 64)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--save-every', type = int, default = 200)
    
    parser.add_argument('--wandb', action = 'store_true', help = 'Whether to use wandb for logging. If enabled, you should provide key, entity and project arguments.')
    parser.add_argument('--wandb_key', type = str, default = 'YOUR_KEY_HERE', help = "Wandb key.")
    parser.add_argument('--wandb_entity', type = str, default = 'YOUR_ENTITY_HERE', help = 'Wandb entity.')
    parser.add_argument('--wandb_project', type = str, default = 'YOUR_PROJECT_HERE', help = 'Wandb project.')

    args = parser.parse_args()
    main(args)