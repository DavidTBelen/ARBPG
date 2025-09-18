# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:26:54 2025

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper:  Randomized block proximal method with locally
Lipschitz continuous gradient
"""



import numpy as np
import matplotlib.pyplot as plt
import os
from Algorithms import NMF_adaptive_R1 as NMF
from numpy import random




theSeed = 1

print('Seed: ',theSeed)
random.seed(theSeed)


def bresenham_line(x0, y0, x1, y1):
    """Return list of points on a line using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def draw_limb(img, joint, angle_deg, length=5, offset=0):
    """Draw a limb starting at joint with absolute angle in degrees (0° = right)."""
    x0, y0 = joint
    x0 += offset
    theta = np.deg2rad(angle_deg)
    x1 = int(round(x0 + length * np.cos(theta)))
    y1 = int(round(y0 - length * np.sin(theta)))  # y decreases upward
    for (x, y) in bresenham_line(x0, y0, x1, y1):
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            img[y, x] = 1

def generate_swimmer_angles(img_size=20, limb_length=5, save_folder=None,
                            make_collage=True, collage_shape=(8, 32)):
    """Generate 256 swimmer images with absolute angles per limb, save them and a collage."""
    H = W = img_size
    torso_top = 6
    torso_bottom = 14
    torso_x = W // 2

    # Absolute angles w.r.t positive x-axis
    angles_left_arm  = [90, 135, 180, 225]
    angles_right_arm = [90, 45, 0, -45]
    angles_left_leg  = [135, 180, 225, 270]
    angles_right_leg = [45, 0, -45, 270]

    left_offset = -1
    right_offset = 1

    images = []
    configs = []

    for la in angles_left_arm:
        for ra in angles_right_arm:
            for ll in angles_left_leg:
                for rl in angles_right_leg:
                    img = np.zeros((H, W), dtype=float)

                    # vertical torso
                    for y in range(torso_top, torso_bottom):
                        img[y, torso_x] = 1

                    # joints
                    shoulder_y = torso_top
                    hip_y = torso_bottom - 1
                    left_arm_joint = (torso_x, shoulder_y)
                    right_arm_joint = (torso_x, shoulder_y)
                    left_leg_joint = (torso_x, hip_y)
                    right_leg_joint = (torso_x, hip_y)

                    # draw limbs
                    draw_limb(img, left_arm_joint, la, limb_length, offset=left_offset)
                    draw_limb(img, right_arm_joint, ra, limb_length, offset=right_offset)
                    draw_limb(img, left_leg_joint, ll, limb_length, offset=left_offset)
                    draw_limb(img, right_leg_joint, rl, limb_length, offset=right_offset)

                    images.append(img)
                    configs.append((la, ra, ll, rl))

    images = np.array(images)  # shape (256, H, W)
    X = images.reshape(len(images), -1).T

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        # save individual images
        for i, img in enumerate(images):
            plt.imsave(os.path.join(save_folder, f"swimmer_{i+1:03d}.jpg"),
                       img, cmap="gray", vmin=0, vmax=1)

        # save collage if requested
        if make_collage:
            rows, cols = collage_shape
            fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
            for i, ax in enumerate(axes.ravel()):
                if i < len(images):
                    ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            collage_path = os.path.join(save_folder, "swimmer_collage.jpg")
            plt.savefig(collage_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    return X, images, configs

X, images, configs = generate_swimmer_angles()

print("Number of images:", images.shape[0], "Data matrix shape:", X.shape)

# Preview first 16 images
fig, axes = plt.subplots(2, 8, figsize=(12,3))
for ax, img, cfg in zip(axes.ravel(), images[:16], configs[:16]):
    ax.imshow(img, cmap="gray")
    ax.set_title(cfg)
    ax.axis("off")
plt.show()


print("\n### Start of the experiment ### \n")

m, n = X.shape

"Algorithmic parameters"
beta = .9
tau_min = 10**(-8)
tau_max = 10**8
sigma = 10**(-4)

# Stopping parameters
gap = 2*(m+n)
prec = 1e-4
stop_rule = 0 
max_stop = 500000 

btau = 2

# Number of basis images:
r = 16


inst = {}

inst['A'] = X
inst['m'] = m
inst['n'] = n
inst['r'] = r

U0 = np.random.random([r,m])
V0 = np.random.random([r,n])

sparse = m - int(np.round(.9*m))


for i in range(r):
    idx = np.argpartition(U0[i, :], sparse)[:sparse]  # indices of s smallest in row i
    U0[i, idx] = 0
    
    




"Algorithm"


sol = NMF.RNBPG_l0(U0,V0,inst,btau,tau_min,tau_max,sigma,beta,sparse,gap,prec,stop_rule=0,max_stop=max_stop) 
print("\n Output algorithm:")
print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
print('Fval=',sol['F'])
print("f eval=",sol['f_eval'])


U = sol['U']
V = sol['V']




import numpy as np
import matplotlib.pyplot as plt


def plot_and_save_basis_collage(U, img_shape=(20, 20), grid_shape=(4, 4), cmap="gray", filename=None, spacing=2):
    """
    Display and optionally save a collage of basis images stored in the rows of U.

    Parameters:
        U : np.ndarray
            Matrix of size (num_basis, num_pixels). Each row is a basis image.
        img_shape : tuple
            Original image shape (height, width).
        grid_shape : tuple
            Grid shape (rows, cols) for the collage.
        cmap : str
            Colormap for display.
        filename : str or None
            If provided, the collage will be saved to this path.
        spacing : int
            Number of pixels between images when saving.
    """
    num_basis = U.shape[0]
    rows, cols = grid_shape
    H, W = img_shape

    # Normalize each row for display
    U_norm = np.zeros_like(U, dtype=float)
    for i in range(num_basis):
        row = U[i, :]
        min_val, max_val = row.min(), row.max()
        U_norm[i, :] = (row - min_val) / (max_val - min_val) if max_val > min_val else row

    # Plotting using matplotlib subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i, ax in enumerate(axes.ravel()):
        if i < num_basis:
            img = U_norm[i, :].reshape(H, W)
            ax.imshow(img, cmap=cmap)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


plot_and_save_basis_collage(U, img_shape=(20,20), grid_shape=(4,4), cmap="gray")


