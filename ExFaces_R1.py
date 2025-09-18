# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:35:02 2025

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper:  Randomized block proximal method with locally
Lipschitz continuous 

IMPORTANT:
This code requires the Database of Faces from the AT&T Laboratories Cambridge.
Go to: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
"""


import os
from Algorithms import NMF_adaptive_R1 as NMF
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




theSeed = 1

print('Seed: ',theSeed)
random.seed(theSeed)


def load_and_preprocess_orl_faces(dataset_path="AT&TFaces",
                                   crop_size=(64, 64),
                                   vertical_offset=10,
                                   save_preprocessed=True,
                                   preprocessed_folder="AT&TFaces_preprocessed"):
    """
    Load ORL faces, preprocess by cropping a rectangle of crop_size starting some pixels below,
    normalize, and optionally save as JPG.

    Parameters:
        dataset_path : str
            Path to original ORL dataset.
        crop_size : tuple
            (height, width) of crop, e.g., (64, 64)
        vertical_offset : int
            Number of pixels to shift crop downward from top-center.
        save_preprocessed : bool
            If True, save preprocessed images.
        preprocessed_folder : str
            Folder to save preprocessed images (same structure as original).

    Returns:
        X : np.ndarray, shape (n_pixels, n_images)
        y : np.ndarray, labels
        image_shape : tuple, (H, W)
    """
    images = []
    labels = []

    crop_h, crop_w = crop_size

    for subject_id in range(1, 41):
        subject_folder = os.path.join(dataset_path, f"s{subject_id}")
        if not os.path.isdir(subject_folder):
            continue

        if save_preprocessed:
            preproc_subject_folder = os.path.join(preprocessed_folder, f"s{subject_id}")
            os.makedirs(preproc_subject_folder, exist_ok=True)

        img_names = sorted([f for f in os.listdir(subject_folder) if f.endswith(".pgm")])
        for img_name in img_names:
            img_path = os.path.join(subject_folder, img_name)
            img = Image.open(img_path).convert("L")  # grayscale

            w, h = img.size

            # Crop rectangle with vertical offset
            left = max((w - crop_w)//2, 0)
            upper = max((h - crop_h)//2 + vertical_offset, 0)
            right = left + crop_w
            lower = upper + crop_h
            img_cropped = img.crop((left, upper, right, lower))

            # Convert to numpy and normalize
            img_array = np.array(img_cropped, dtype=float) / 255.0

            # Flatten
            images.append(img_array.flatten())
            labels.append(subject_id - 1)

            # Save preprocessed image as JPG
            if save_preprocessed:
                save_name = os.path.splitext(img_name)[0] + ".jpg"
                save_path = os.path.join(preproc_subject_folder, save_name)
                img_to_save = Image.fromarray(np.uint8(img_array*255))
                img_to_save.save(save_path)

    X = np.array(images).T
    y = np.array(labels)
    image_shape = crop_size

    return X, y, image_shape



def plot_faces_collage(X, image_shape=(64, 64), grid_shape=(10, 40), cmap="gray", filename="faces_collage.jpg"):
    """
    Plot and save a collage of preprocessed faces with no blank space.

    Parameters:
        X : np.ndarray
            Data matrix of shape (n_pixels, n_images). Each column is a face.
        image_shape : tuple
            Shape of each face image (H, W).
        grid_shape : tuple
            (rows, cols) of the collage grid.
        cmap : str
            Colormap for displaying images.
        filename : str
            Path to save the collage image.
    """
    H, W = image_shape
    rows, cols = grid_shape
    num_images = rows * cols

    if X.shape[1] < num_images:
        raise ValueError(f"Not enough images: have {X.shape[1]}, need {num_images}")

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.ravel()):
        img = X[:, i].reshape(H, W)
        ax.imshow(img, cmap=cmap)
        ax.axis("off")

    # Remove all spacing, margins, and borders
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Collage saved to {filename}")


# Example
X, y, img_shape = load_and_preprocess_orl_faces("AT&TFaces")

plot_faces_collage(X, image_shape=img_shape, grid_shape=(10, 40))


X = X.astype(float)/255
print("Data loaded: \n")
print("Data matrix shape:", X.shape)
print("Number of subjects:", len(np.unique(y)))
print("Each image shape:", img_shape)


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
max_stop =500000 

btau = 2

# Number of basis images:
r = 25


inst = {}

inst['A'] = X
inst['m'] = m
inst['n'] = n
inst['r'] = r

U0 = np.random.random([r,m])
V0 = np.random.random([r,n])

factor = .33
sparse = m - int(np.round(factor*m))


for i in range(r):
    idx = np.argpartition(U0[i, :], sparse)[:sparse]  # indices of s smallest in row i
    U0[i, idx] = 0
    
    




"Algorithm"

sol = NMF.RNBPG_efficient_l0(U0,V0,inst,btau,tau_min,tau_max,sigma,beta,sparse,gap,prec,stop_rule=0,max_stop=max_stop) 
print("\n Output algorithm:")
print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
print('Fval=',sol['F'])
print("f eval=",sol['f_eval'])



    
U = sol['U']
V = sol['V']




def normalize_basis(U):
    U_norm = np.zeros_like(U, dtype=float)
    for i in range(U.shape[0]):
        row = U[i, :]
        min_val = row.min()
        max_val = row.max()
        if max_val > min_val:
            U_norm[i, :] = (row - min_val) / (max_val - min_val)
        else:
            U_norm[i, :] = row  # if constant, leave unchanged
    return U_norm

# Example usage
U_normalized = normalize_basis(U)



def save_basis_collage(U, img_shape=(20, 20), grid_shape=(4, 4), spacing=2, filename="collage.jpg", cmap="gray"):
    """
    Save a collage of basis images (rows of U) as a single image.
    """
    U_norm = np.zeros_like(U, dtype=float)
    for i in range(U.shape[0]):  # normalize each row
        row = U[i, :]
        min_val = row.min()
        max_val = row.max()
        U_norm[i, :] = (row - min_val) / (max_val - min_val) if max_val > min_val else row

    H, W = img_shape
    rows, cols = grid_shape

    # Create empty canvas
    collage_height = rows * H + (rows - 1) * spacing
    collage_width = cols * W + (cols - 1) * spacing
    collage = np.zeros((collage_height, collage_width))

    # Fill canvas with images
    for idx in range(U.shape[0]):
        r = idx // cols
        c = idx % cols
        top = r * (H + spacing)
        left = c * (W + spacing)
        collage[top:top+H, left:left+W] = U_norm[idx, :].reshape(H, W)

    # Save the collage
    plt.imsave(filename, collage, cmap=cmap, vmin=0, vmax=1)
    print(f"Collage saved as '{filename}'")
 
save_basis_collage(U, img_shape=img_shape, grid_shape=(5,5))



