from mpmath import *
import matplotlib.pyplot as plt
import numpy as np
import math

def cross(a: list, b: list):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

def dot(a: list, b: list):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def norm(a: list):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def add(a: list, b: list):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

def subtract(a: list, b: list):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

def multiply(a: list, b: float):
    return [a[0] * b, a[1] * b, a[2] * b]

def divide(a: list, b: float):
    return [a[0] / b, a[1] / b, a[2] / b]

mp.dps = 1000

# Overall relative error, theta -> 0
v1_true = [mpf('1.0'), mpf('0.0'), mpf('0.0')]
v1_approx = [1.0, 0.0, 0.0]

x = [power(mpf(10), -(mpf(i) + mpf('0.01') * j)) for i in range(1, 8) for j in range(0, 100)]
rel_err_1 = []
rel_err_2 = []
rel_err_3 = []
rel_err_4 = []

for i in range(1, 8):
    for j in range(0, 100):
        v2_true = [cos(power(mpf(10), -(mpf(i) + mpf('0.01') * j))), sin(power(mpf(10), -(mpf(i) + mpf('0.01') * j))), mpf(0.0)]
        v2_approx = [math.cos(math.pow(10, -(i + 0.01 * j))), math.sin(math.pow(10, -(i + 0.01 * j))), 0.0]
        theta_true = power(mpf(10), -(mpf(i) + mpf('0.01') * j))

        # method 1
        theta_approx = math.acos(dot(v1_approx, v2_approx) / (norm(v1_approx) * norm(v2_approx)))
        rel_err_1.append(fabs(theta_true - theta_approx) / theta_true)

        # method 2
        theta_approx = math.atan2(norm(cross(v1_approx, v2_approx)), dot(v1_approx, v2_approx))
        rel_err_2.append(fabs(theta_true - theta_approx) / theta_true)

        # method 3
        norm_v1_approx = norm(v1_approx)
        norm_v2_approx = norm(v2_approx)
        theta_approx = 2 * math.atan2(norm(subtract(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx))), norm(add(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx))))
        rel_err_3.append(fabs(theta_true - theta_approx) / theta_true)

        # method 4
        theta_approx = 2 * math.atan2(norm(subtract(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx))), norm(add(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx))))
        rel_err_4.append(fabs(theta_true - theta_approx) / theta_true)

fig, (ax) = plt.subplots(1, 1)
ax.plot(x, rel_err_1, label='Method 1', lw=1)
ax.plot(x, rel_err_2, label='Method 2', lw=1)
ax.plot(x, rel_err_3, label='Method 3', lw=1)
ax.plot(x, rel_err_4, label='Method 4', lw=1)
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlabel(r'Angle $\theta$')
ax.set_ylabel('Relative Error')
ax.legend()
fig.savefig("overall_relative_error_theta_0.png")

# Overall relative error, theta -> pi/2
v1_true = [mpf('1.0'), mpf('0.0'), mpf('0.0')]
v1_approx = [1.0, 0.0, 0.0]

x = [power(mpf(10), -(mpf(i) + mpf('0.01') * j)) for i in range(1, 8) for j in range(0, 100)]
rel_err_1 = []
rel_err_2 = []
rel_err_3 = []
rel_err_4 = []

for i in range(1, 8):
    for j in range(0, 100):
        v2_true = [cos(pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))), sin(pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))), mpf(0.0)]
        v2_approx = [math.cos(math.pi / 2 - math.pow(mpf(10), -(mpf(i) + mpf('0.01') * j))), math.sin(math.pi / 2 - math.pow(mpf(10), -(mpf(i) + mpf('0.01') * j))), 0.0]
        theta_true = pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))

        # method 1
        theta_approx = math.acos(dot(v1_approx, v2_approx) / (norm(v1_approx) * norm(v2_approx)))
        rel_err_1.append(fabs(theta_true - theta_approx) / theta_true)

        # method 2
        theta_approx = math.atan2(norm(cross(v1_approx, v2_approx)), dot(v1_approx, v2_approx))
        rel_err_2.append(fabs(theta_true - theta_approx) / theta_true)

        # method 3
        norm_v1_approx = norm(v1_approx)
        norm_v2_approx = norm(v2_approx)
        theta_approx = 2 * math.atan2(norm(subtract(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx))), norm(add(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx))))
        rel_err_3.append(fabs(theta_true - theta_approx) / theta_true)

        # method 4
        theta_approx = 2 * math.atan2(norm(subtract(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx))), norm(add(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx))))
        rel_err_4.append(fabs(theta_true - theta_approx) / theta_true)

fig, (ax) = plt.subplots(1, 1)
ax.plot(x, rel_err_1, label='Method 1', lw=1)
ax.plot(x, rel_err_2, label='Method 2', lw=1)
ax.plot(x, rel_err_3, label='Method 3', lw=1)
ax.plot(x, rel_err_4, label='Method 4', lw=1)
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlabel(r'Angle $\theta$ ($\pi/2-x$)')
ax.set_ylabel('Relative Error')
ax.legend()
fig.savefig("overall_relative_error_theta_90.png")


# Instrinsic relative error, theta -> 0
v1_true = [mpf('1.0'), mpf('0.0'), mpf('0.0')]
v1_approx = [1.0, 0.0, 0.0]

x = [power(mpf(10), -(mpf(i) + mpf('0.01') * mpf(j))) for i in range(1, 8) for j in range(0, 100)]
rel_err_1 = []
rel_err_2 = []
rel_err_3 = []
rel_err_4 = []

for i in range(1, 8):
    for j in range(0, 100):
        v2_true = [cos(power(mpf(10), -(mpf(i) + mpf('0.01') * mpf(j)))), sin(power(mpf(10), -(mpf(i) + mpf('0.01') * mpf(j)))), mpf(0.0)]
        v2_approx = [math.cos(math.pow(10, -(i + 0.01 * j))), math.sin(math.pow(10, -(i + 0.01 * j))), 0.0]
        
        # method 1
        theta_true = fdot(v1_true, v2_true)
        theta_approx = dot(v1_approx, v2_approx) / (norm(v1_approx) * norm(v2_approx))
        rel_err_1.append(fabs(theta_true - theta_approx) / theta_true)

        # method 2
        theta_true = tan(power(mpf(10), -(mpf(i) + mpf('0.01') * mpf(j))))
        theta_approx = norm(cross(v1_approx, v2_approx)) / dot(v1_approx, v2_approx)
        rel_err_2.append(fabs(theta_true - theta_approx) / theta_true)

        # method 3
        theta_true = tan(power(mpf(10), -(mpf(i) + mpf('0.01') * mpf(j))) / mpf(2))
        norm_v1_approx = norm(v1_approx)
        norm_v2_approx = norm(v2_approx)
        theta_approx = norm(subtract(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx))) / norm(add(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx)))
        rel_err_3.append(fabs(theta_true - theta_approx) / theta_true)

        # method 4
        theta_true = tan(power(mpf(10), -(mpf(i) + mpf('0.01') * mpf(j))) / mpf(2))
        theta_approx = norm(subtract(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx))) / norm(add(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx)))
        rel_err_4.append(fabs(theta_true - theta_approx) / theta_true)

fig, (ax) = plt.subplots(1, 1)
ax.plot(x, rel_err_1, label='Method 1', lw=1)
ax.plot(x, rel_err_2, label='Method 2', lw=1)
ax.plot(x, rel_err_3, label='Method 3', lw=1)
ax.plot(x, rel_err_4, label='Method 4', lw=1)
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlabel(r'Angle $\theta$')
ax.set_ylabel('Intrinsic Relative Error')
ax.legend()
fig.savefig("intrinsic_relative_error_theta_0.png")

# Intrinsic relative error, theta -> pi/2
v1_true = [mpf('1.0'), mpf('0.0'), mpf('0.0')]
v1_approx = [1.0, 0.0, 0.0]

x = [power(mpf(10), -(mpf(i) + mpf('0.01') * j)) for i in range(1, 8) for j in range(0, 100)]
rel_err_1 = []
rel_err_2 = []
rel_err_3 = []
rel_err_4 = []

for i in range(1, 8):
    for j in range(0, 100):
        v2_true = [cos(pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))), sin(pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))), mpf(0.0)]
        v2_approx = [math.cos(math.pi / 2 - math.pow(10, -(i + 0.01 * j))), math.sin(math.pi / 2 - math.pow(10, -(i + 0.01 * j))), 0.0]
        
        # method 1
        theta_true = fdot(v1_true, v2_true)
        theta_approx = dot(v1_approx, v2_approx) / (norm(v1_approx) * norm(v2_approx))
        rel_err_1.append(fabs(theta_true - theta_approx) / theta_true)

        # method 2
        theta_true = tan(pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j)))
        theta_approx = norm(cross(v1_approx, v2_approx)) / dot(v1_approx, v2_approx)
        rel_err_2.append(fabs(theta_true - theta_approx) / theta_true)

        # method 3
        theta_true = tan((pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))) / 2)
        norm_v1_approx = norm(v1_approx)
        norm_v2_approx = norm(v2_approx)
        theta_approx = norm(subtract(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx))) / norm(add(divide(v1_approx, norm_v1_approx), divide(v2_approx, norm_v2_approx)))
        rel_err_3.append(fabs(theta_true - theta_approx) / theta_true)

        # method 4
        theta_true = tan((pi / 2 - power(mpf(10), -(mpf(i) + mpf('0.01') * j))) / 2)
        theta_approx = norm(subtract(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx))) / norm(add(multiply(v1_approx, norm_v2_approx), multiply(v2_approx, norm_v1_approx)))
        rel_err_4.append(fabs(theta_true - theta_approx) / theta_true)

fig, (ax) = plt.subplots(1, 1)
ax.plot(x, rel_err_1, label='Method 1', lw=1)
ax.plot(x, rel_err_2, label='Method 2', lw=1)
ax.plot(x, rel_err_3, label='Method 3', lw=1)
ax.plot(x, rel_err_4, label='Method 4', lw=1)
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlabel(r'Angle $\theta$ ($\pi/2-x$)')
ax.set_ylabel('Intrinsic Relative Error')
ax.legend()
fig.savefig("intrinsic_relative_error_theta_90.png")
