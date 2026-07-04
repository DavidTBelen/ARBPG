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
from numpy import random
from time import time
from numpy.linalg import norm
from Auxiliaries import DisplayResultsNMF as DRNMF


"Algorithm"


def ARBPG_l0(U0, V0, inst, tau_min, tau_max,btau, sigma, beta, sparse, gap=50, prec=1e-6, stop_rule=0, max_stop=1000):
    A = inst['A']
    r = inst['r']
    
    normA = norm(A)
    
    U = U0.copy()
    V = V0.copy()
    
    f_eval = 0
    AUV = A - U.T @ V
    
    F_old = 0.5 * norm(AUV, 'fro')**2
    F_list = [F_old]  
    
    counter = [0, 0]
    t0 = time()
    
    variables_updated = 0 
    total_variables = (r*U.shape[1]) + (r*V.shape[1])
    
    continua = True
    it = 0

    
    
    while continua:
        ik = random.randint(0, 2 * r)
        
       
        # ---------------------------------------------------------------------
        # Update U
        # ---------------------------------------------------------------------
        if ik < r:
            variables_updated += U.shape[1] #

            ui = U[ik, :]
            Vik = V[ik, :]
            
            tauU = btau / (np.dot(Vik, Vik) + 2 * sigma)
            tauU = max(min(tau_max, tauU), tau_min)
            
            gf = -Vik @ AUV.T
            
            adaptive_ls = True
            while adaptive_ls:
                temp_ui = ui - tauU * gf
                sparse_ui = np.maximum(temp_ui, 0.)
                
                idui = np.argpartition(sparse_ui, sparse)[:sparse]
                sparse_ui[idui] = 0.0
                
                di = -ui + sparse_ui
                di_norm_sq = np.dot(di, di)
                
                if di_norm_sq > 1e-14:
                    temp_arg = 0.5 * di_norm_sq * np.dot(Vik, Vik) + np.dot(gf, di)
                    f_eval += 1
                    
                    if temp_arg <= -sigma * di_norm_sq:
                        U[ik, :] += di
                        AUV -= np.outer(di, Vik)  
                        
                        F_new = F_old + temp_arg
                        F_list.append(F_new)
                        F_old = F_new
                        adaptive_ls = False
                    else:
                        tauU = beta * tauU
                else:
                    F_list.append(F_old)
                    break
        # ---------------------------------------------------------------------
        # Update V
        # ---------------------------------------------------------------------
        else:
            variables_updated += V.shape[1]

            jk = ik - r
            vi = V[jk, :]
            Uik = U[jk, :]
            
            tauV = btau / (np.dot(Uik, Uik) + 2 * sigma)
            tauV = max(min(tau_max, tauV), tau_min)
            
            gf = -Uik @ AUV
            
            adaptive_ls = True
            while adaptive_ls:
                temp_vi = vi - tauV * gf
                di = -vi + np.maximum(temp_vi, 0.)
                di_norm_sq = np.dot(di, di)
                
                if di_norm_sq > 1e-14:
                    temp_arg = 0.5 * di_norm_sq * np.dot(Uik, Uik) + np.dot(gf, di)
                    f_eval += 1
                    
                    if temp_arg <= -sigma * di_norm_sq:
                        V[jk, :] += di
                        AUV -= np.outer(Uik, di) 
                        
                        F_new = F_old + temp_arg
                        F_list.append(F_new)
                        F_old = F_new
                        adaptive_ls = False
                    else:
                        tauV = beta * tauV
                else:
                    F_list.append(F_old)
                    break


         
        # ---------------------------------------------------------------------
        # Check stopping condition
        # ---------------------------------------------------------------------
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap, it + 1)] - F_old) / normA
        
        if counter[0] % 100000 == 0:
            print(f"Iteration: {counter[0]}")
            
        if stop_rule < 2:
            if error_crit <= prec or counter[stop_rule] >= max_stop:
                continua = False
        else:
            if counter[0] >= 1500000 or F_old <= max_stop:
                continua = False

    solution = {
        'U': U,
        'V': V,
        'F': F_old,
        'F_list': np.array(F_list),  
        'it': counter[0],
        'time': counter[1],
        'f_eval': f_eval,
        'epochs': variables_updated/total_variables

    }
    
    return solution


def ipalm_nmf(A,U0,V0, r, sparse, max_iter=1000, tol=1e-5, alpha=0.2, beta=0.2):
  
    m, n = A.shape
    normA = norm(A)
    
    U = U0.copy() 
    V = V0.copy()
    
    def project_U(Mat, s_count):
        Mat_proj = np.maximum(Mat, 0.0)
        if s_count > 0:
            for i in range(Mat_proj.shape[0]):
                idx = np.argpartition(Mat_proj[i], s_count)[:s_count]
                Mat_proj[i, idx] = 0
        return Mat_proj

    U = project_U(U, sparse)
    V = np.maximum(V, 0)
    
    U_old = U.copy()
    V_old = V.copy()
    
    AT = A.T
    norm_A_sq = np.sum(A**2)
    obj_history = []
    
    t0 = time()
    
    V_VT = V @ V.T
    
    for it in range(max_iter):
        # ------------------------------------
        # 1. BLOCK UPDATE FOR U
        # ------------------------------------
        U_ext1 = U + alpha * (U - U_old)
        U_ext2 = U + beta * (U - U_old)
        U_old = U.copy() 
        
        V_AT = V @ AT
        
        grad_U = V_VT @ U_ext2 - V_AT
        
        L_U = np.linalg.norm(V_VT, 2)
        tau = 0.95 / max(L_U, 1e-9)
        
        U = U_ext1 - tau * grad_U
        U = project_U(U, sparse)
        
        # ------------------------------------
        # 2. BLOCK UPDATE FOR V
        # ------------------------------------
        V_ext1 = V + alpha * (V - V_old)
        V_ext2 = V + beta * (V - V_old)
        V_old = V.copy() 
        
        U_UT = U @ U.T
        U_A = U @ A
        
        grad_V = U_UT @ V_ext2 - U_A
        
        L_V = np.linalg.norm(U_UT, 2)
        sigma = 0.95 / max(L_V, 1e-9)
        
        V = np.maximum(V_ext1 - sigma * grad_V, 0)
            
        V_VT = V @ V.T
        
        obj = 0.5 * (norm_A_sq - 2.0 * np.sum(U_A * V) + np.sum(U_UT * V_VT))
        obj_history.append(obj)
        
        final_time = time()-t0
        if it > 0:
            rel_change = abs(obj_history[-2] - obj) / normA
            if rel_change < tol:
                break
    else:
        print(f"Reached max iterations ({max_iter}). Final Objective: {obj_history[-1]:.4f}")
        
    solution = {
        'U': U,
        'V': V,
        'F': obj_history[-1],
        'F_list': obj_history,  
        'it': it,
        'time': final_time,
        'f_eval': 0
    }
                
    return solution


def ARBPG_2blocks(U0,V0, inst,tau_min, tau_max,btau,sigma, sparse, gap=50, prec=1e-6,stop_rule =0, max_stop =1000 ):
    """
    ARBPG with 2-blocks
       
    """
    A = inst['A']
    r = inst['r']
    
    normA = norm(A)
    m, n = A.shape
    
    U = U0.copy() 
    V = V0.copy()
    
    def project_U(Mat, s_count):
        Mat_proj = np.maximum(Mat, 0.0)
        if s_count > 0:
            for i in range(Mat_proj.shape[0]):
                idx = np.argpartition(Mat_proj[i], s_count)[:s_count]
                Mat_proj[i, idx] = 0
        return Mat_proj

    U = project_U(U, sparse)
    V = np.maximum(V, 0)
    
    AT = A.T
    norm_A_sq = np.sum(A**2)
    
    f_eval = 0
    AUV = A - U.T @ V
    
    F_old = 0.5 * norm(AUV, 'fro')**2
    F_list = [F_old]  
    
    it = 0 
    counter = [0, 0]  
    
    variables_updated = 0 
    total_variables = (r*U.shape[1]) + (r*V.shape[1])
    variablesU = r*U.shape[1]
    variablesV = r*V.shape[1]

    
    continua = True
    U_UT = U@U.T
    V_VT = V@V.T
    U_A = U@A

    t0 = time()
    while continua:
        ik = random.random(1)
        
        if ik < 0.5:
            variables_updated += variablesU

            
            V_AT = V@AT
            L_U = norm(V_VT,2)
            tauU = btau / (L_U + 2 * sigma)
            tauU = max(min(tau_max, tauU), tau_min)
            
            grad_U = V_VT @U - V_AT
            
            temp_U = U  - tauU * grad_U 
            U = project_U(temp_U,sparse)
            
            U_UT = U@U.T
            U_A = U@A
            
            obj =  0.5 * (norm_A_sq - 2.0 * np.sum(U_A * V) + np.sum(U_UT * V_VT))
            F_list.append(obj)

        else:
            variables_updated += variablesV

            grad_V = U_UT@V-U_A
            L_V = np.linalg.norm(U_UT,2)
            tauV = btau/(L_V + 2*sigma)
            tauV = max(min(tau_max, tauV), tau_min)
            
            V =  np.maximum(V - tauV * grad_V, 0.)
            
            V_VT = V @ V.T 
            
            obj =  0.5 * (norm_A_sq - 2.0 * np.sum(U_A * V) + np.sum(U_UT * V_VT))
            F_list.append(obj)
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap, it + 1)] - F_list[-1]) / normA
        
        if stop_rule < 2:
            if error_crit <= prec or counter[stop_rule] >= max_stop:
                continua = False
        else:
            if counter[0] >= 1500000 or F_old <= max_stop:
                continua = False
                
                
    solution = {
        'U': U,
        'V': V,
        'F': obj,
        'F_list': np.array(F_list),  
        'it': counter[0],
        'time': counter[1],
        'f_eval': f_eval,
        'epochs': variables_updated/total_variables

    }
    
    return solution
        

    

"Auxiliary functions"


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
    if filename != None:
        collage_path = os.path.join("../FiguresR2", filename)
        plt.savefig(collage_path, dpi=300, bbox_inches="tight")
    plt.show()



X, images, configs = generate_swimmer_angles()

print("Number of images:", images.shape[0], "Data matrix shape:", X.shape)

# Preview first 16 images
fig, axes = plt.subplots(2, 8, figsize=(12,3))
for ax, img, cfg in zip(axes.ravel(), images[:16], configs[:16]):
    ax.imshow(img, cmap="gray")
    ax.set_title(cfg)
    ax.axis("off")
plt.show()



"Experiment"
print("\n### Start of the experiment ### \n")

run_experiment = True
if run_experiment==True:
    theSeed = 1 #random.randint(0,10000)
    
    print('Seed: ',theSeed)
    random.seed(theSeed)
    
    m, n = X.shape
    
    "Algorithmic parameters"
    beta = .9
    tau_min = 10**(-8)
    tau_max = 10**8
    sigma = 10**(-4)
    
    # Stopping parameters
    prec = 1e-8
    stop_rule = 0 
    max_stop = 10000
    
    
    # Number of basis images:
    r = 16
    gap = 2*r
    
      
    inst = {}
    
    inst['A'] = X
    inst['m'] = m
    inst['n'] = n
    inst['r'] = r
    
    
    
    sparse =  m - int(np.round(.33*m))
    
    
    rep = 10
    output_iterations = np.zeros([rep,3])
    output_epochs = np.zeros([rep,3])
    output_time = np.zeros([rep,3])
    output_Fval = np.zeros([rep,3])
    
    
    "Algorithm"
    
    for rreepp in range(rep):
        
        U0 = np.random.random([r,m])
        V0 = np.random.random([r,n])
        
        for i in range(r):
            idx = np.argpartition(U0[i, :], sparse)[:sparse]  
            U0[i, idx] = 0
        
        
        seed_algorithms = random.randint(0,10000)
    
    
        random.seed(seed_algorithms)
        gap = 2*r
        sol = ARBPG_l0(U0,V0,inst,tau_min,tau_max,.95,sigma,beta,sparse,gap,prec,stop_rule=0,max_stop=max_stop) 
        print("\n ARBPG:")
        print(np.round(sol['time'],4), 'seconds ',sol['epochs'], 'epochs')
        print('Fval=',sol['F'])
        print("f eval=",sol['f_eval'])
        
        
        U1 = sol['U']
        V1 = sol['V']
        
        output_iterations[rreepp,0] = sol['it']
        output_epochs[rreepp,0] = sol['epochs']
        output_time[rreepp,0] = sol['time']
        output_Fval[rreepp,0] = sol['F']
    
        
        gap = 2 
        random.seed(seed_algorithms)
        sol = ARBPG_2blocks(U0,V0, inst,tau_min, tau_max,0.95,sigma, sparse, gap=gap, prec=prec,stop_rule =0, max_stop=max_stop)
        print("\n ARBPG Full block:")
        print(np.round(sol['time'],4), 'seconds ',sol['epochs'], 'epochs')
        print('Fval=',sol['F'])
        print("f eval=",sol['f_eval'])
        
        output_iterations[rreepp,1] = sol['it']
        output_epochs[rreepp,1] = sol['epochs']
        output_time[rreepp,1] = sol['time']
        output_Fval[rreepp,1] = sol['F']
    
    
        UB = sol['U']
        VB = sol['V']
        
        solP = ipalm_nmf(X,U0,V0, r,sparse,  max_iter=max_stop, tol=prec, alpha=0.2, beta=0.2)
        print('\n  iPALM 0.2:')
        print(np.round(solP['time'],4), 'seconds ',solP['it'], 'epochs')
        print('Fval=',solP['F'])
        print("f eval=",solP['f_eval'])
    
        
        output_iterations[rreepp,2] = solP['it']
        output_epochs[rreepp,2] = solP['it']
        output_time[rreepp,2] = solP['time']
        output_Fval[rreepp,2] = solP['F']
        
        UP = solP['U']
        VP = solP['V']
        
        
        



import matplotlib.pyplot as plt





filename10 = "collage_fullblocks.jpg"

plot_and_save_basis_collage(UB, img_shape=(20,20), grid_shape=(4,4), cmap="gray", filename=filename10)

filenameP = "collage_palm.jpg"

plot_and_save_basis_collage(UP, img_shape=(20,20), grid_shape=(4,4), cmap="gray", filename=filenameP)

