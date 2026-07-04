# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:49:52 2026

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper: Randomized block proximal method with locally
Lipschitz continuous gradient
"""

import numpy as np
from numpy import random
from PIL import Image
import matplotlib.pyplot as plt
from Auxiliaries import DisplayResultsNMF as DRNMF
from time import time
from numpy.linalg import norm

from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _initialize_nmf

" Algorithms "


"ARBPG"
def ARBPG(U0, V0, inst, tau_min, tau_max, sigma, gap=50, prec=1e-6, stop_rule=0, max_stop=1000):
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
    
    continua = True
    it = 0
    
    variables_updated = 0 
    total_variables = (r*U.shape[1]) + (r*V.shape[1])
    
    while continua:
        ik = random.randint(0, 2 * r)
        # ---------------------------------------------------------------------
        # Update U
        # ---------------------------------------------------------------------
        if ik < r:
            variables_updated += U.shape[1] 
            ui = U[ik, :].copy()
            Vik = V[ik, :].copy()
            
            tauU = .95/ (np.dot(Vik, Vik) + 2 * sigma)
            tauU = max(min(tau_max, tauU), tau_min)
            
            gf = -Vik @ AUV.T
            
            temp_ui = ui - tauU * gf
            di = -ui + np.maximum(temp_ui, 0.)
            di_norm_sq = np.dot(di, di)
            
            if di_norm_sq > 1e-14:
                temp_arg = 0.5 * di_norm_sq * np.dot(Vik, Vik) + np.dot(gf, di)
            
                U[ik, :] += di
                AUV -= np.outer(di, Vik)  
                
                F_new = F_old + temp_arg
                F_list.append(F_new)
                F_old = F_new
                
            else:
                F_list.append(F_old)
        # ---------------------------------------------------------------------
        # Update V
        # ---------------------------------------------------------------------
        else:
            variables_updated += V.shape[1]
            jk = ik - r
            vi = V[jk, :].copy()
            Uik = U[jk, :].copy()
            
            tauV = .95 / (np.dot(Uik, Uik) + 2 * sigma)
            tauV = max(min(tau_max, tauV), tau_min)
            
            gf = -Uik @ AUV
            
            temp_vi = vi - tauV * gf
            di = -vi + np.maximum(temp_vi, 0.)
            di_norm_sq = np.dot(di, di)
                
            if di_norm_sq > 1e-14:
                temp_arg = 0.5 * di_norm_sq * np.dot(Uik, Uik) + np.dot(gf, di)
                
                V[jk, :] += di
                AUV -= np.outer(Uik, di) 
                
                F_new = F_old + temp_arg
                F_list.append(F_new)
                F_old = F_new

            else:
                # Direction is zero
                F_list.append(F_old)

         
        # ---------------------------------------------------------------------
        # Check stopping condition
        # ---------------------------------------------------------------------
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap, it + 1)] - F_old) / normA
        
 
            
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



def ARBPG_block(U0, V0, inst, tau_min, tau_max, sigma, block_size=10, gap=50, prec=1e-6, stop_rule=0, max_stop=1000):
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
    
    continua = True
    it = 0

    blocks = []
    for start in range(0, r, block_size):
        end = min(start + block_size, r)
        blocks.append(('U', start, end))
        blocks.append(('V', start, end))
    
    num_blocks = len(blocks)
    
    variables_updated = 0 
    total_variables = (r * U.shape[1]) + (r * V.shape[1])
    while continua:
        b_idx = random.randint(0, num_blocks - 1)
        block_type, start, end = blocks[b_idx]
        
        # ---------------------------------------------------------------------
        # Update Block of U
        # ---------------------------------------------------------------------
        if block_type == 'U':
            variables_updated += (end - start) * U.shape[1]
            
            U_block = U[start:end, :].copy()      
            V_block = V[start:end, :].copy()     
            
            V_sub_prod = V_block @ V_block.T  
            L_block = norm(V_sub_prod, 2)   
            
            tauU = 0.95 / (L_block + 2 * sigma)
            tauU = max(min(tau_max, tauU), tau_min)
            
            gf = -V_block @ AUV.T          
            
            
            temp_U = U_block - tauU * gf
            D = -U_block + np.maximum(temp_U, 0.) 
            D_norm_sq = norm(D, 'fro')**2
            
            if D_norm_sq > 1e-14:
                DD_t = D @ D.T
                quad_term = 0.5 * np.sum(DD_t * V_sub_prod)
                lin_term = np.sum(gf * D) 
                temp_arg = quad_term + lin_term
                
                
                U[start:end, :] += D
                AUV -= D.T @ V_block  
                
                F_new = F_old + temp_arg
                F_list.append(F_new)
                F_old = F_new
            else:
                F_list.append(F_old)
                    
        # ---------------------------------------------------------------------
        # Update Block of V
        # ---------------------------------------------------------------------
        else:
            variables_updated += (end - start) * V.shape[1]
            
            V_block = V[start:end, :].copy()      
            U_block = U[start:end, :].copy()      
            
            U_sub_prod = U_block @ U_block.T  
            L_block = norm(U_sub_prod, 2)
            
            tauV = 0.95 / (L_block + 2 * sigma)
            tauV = max(min(tau_max, tauV), tau_min)
            
            gf = -U_block @ AUV          
            

            temp_V = V_block - tauV * gf
            D = -V_block + np.maximum(temp_V, 0.) 
            D_norm_sq = norm(D, 'fro')**2
            
            if D_norm_sq > 1e-14:
                DD_t = D@D.T
                quad_term = 0.5 * np.sum(DD_t*U_sub_prod)
                lin_term = np.sum(gf * D)
                temp_arg = quad_term + lin_term
                f_eval += 1
                
                
                V[start:end, :] += D
                AUV -= U_block.T @ D   
                
                F_new = F_old + temp_arg
                F_list.append(F_new)
                F_old = F_new
            else:
                F_list.append(F_old)
            

        # ---------------------------------------------------------------------
        # Check stopping condition
        # ---------------------------------------------------------------------
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap, it + 1)] - F_old) / normA
        

            
        if stop_rule < 2:
            if (variables_updated / total_variables) > 100:
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
        'epochs': variables_updated / total_variables
    }
    
    return solution



"ipalm"
def ipalm_nmf(A,U0,V0, r, max_iter=1000, tol=1e-5, alpha=0.2, beta=0.2):

    m, n = A.shape
    normA = norm(A)
    
    U = U0.copy() 
    V = V0.copy()
    
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
        tau = 0.95 /L_U
        
        U = np.maximum(U_ext1 - tau * grad_U,0)
        
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
        sigma = 0.95 / L_V
        
        V = np.maximum(V_ext1 - sigma * grad_V, 0)
        
        V_VT = V @ V.T
        
        obj = 0.5 * (norm_A_sq - 2.0 * np.sum(U_A * V) + np.sum(U_UT * V_VT))
        obj_history.append(obj)
        
        final_time = time()-t0
        if it > 0:
            rel_change = abs(obj_history[-2] - obj) / normA #max(1.0, obj_history[-2])
            if rel_change < tol:
                break
    else:
        print(f"Reached max iterations ({max_iter}). Final Objective: {obj_history[-1]:.4f}")
    
    solution = {
        'U': U,
        'V': V,
        'F': obj_history[-1],
        'F_list': obj_history,  # Convert to array only once at the end
        'it': it,
        'time': final_time,
        'f_eval': 0
    }
    return solution



" Experiment 1: Comparison of Block sizes with iPALM"
experiment1 = True
if experiment1:
    
    

    theSeed = 1
    
    print('Seed: ',theSeed)
    random.seed(theSeed)
    
    images = [ 'Atacama_color', 'Santiago_color', 'Valdivia_color', 'Niebla_color']
    
    rs = [100, 100, 100, 150]
    block_sizes = [5,10,20]
    
    # Only color red
    rep = 10 # Number of repetitions 
    
    
    # For storing the data:
    output_epochs = np.zeros([len(images),rep,6]) # images, rep, algs
    output_time = np.zeros([len(images),rep,6])
    output_Fval = np.zeros([len(images),rep,6])
    output_PSNR = np.zeros([len(images),rep,6]) #peak signal-to-noise ratio
    
    
    
    
    for ii in range(len(images)):
        
        r = rs[ii] # rank of the compression
        
        sel_image = images[ii]
        fig_name = 'Data/'+sel_image+'.jpg'  
        Im = plt.imread(fig_name)
        IR = Im[:,:,0]
        
        IR = IR.astype(float)/255
        
        m, n = IR.shape
        
        "Algorithmic parameters"
        
        # ARBPG parameters:
        tau_min = 10**(-8)
        tau_max = 10**8
        sigma= 10**(-4)
        
        # Stopping parameters
        prec = 1e-4
        stop_rule = 0 
        max_stop = 1500000
        
        
        print('\n ### Image:', sel_image, 'Color:', 'Red')
        
        inst = {}
        
        inst['A'] = IR
        inst['m'] = m
        inst['n'] = n
        inst['r'] = r
        
        for rreepp in range(rep):
            print("\n Repetition:",rreepp)
            
            # Same initialization for all the algorithms
            U0 = np.random.random([r,m])
            V0 = np.random.random([r,n])
            
            "ARBPG:" 
            gap = 2*r
            sol = ARBPG(U0,V0,inst,tau_min,tau_max,sigma,gap,prec,stop_rule=0,max_stop=max_stop)
            print('\n ARBPG:')
            print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
            print("epochs", np.round(sol['epochs'],1))
            print('Fval=',sol['F'])
            
            UN = sol['U']
            VN = sol['V']
            
            IRN = UN.T@VN
            
            
            IRN_im_0 = (255.0 * IRN).astype(np.uint8)
            IRN_im = Image.fromarray(IRN_im_0)
            
            psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRN-IR,'fro')**2))
            print('PSNR=',psnr)
            
            output_epochs[ii,rreepp,0] = np.round(sol['epochs'],1)
            output_time[ii,rreepp,0] = np.round(sol['time'],2)
            output_Fval[ii,rreepp,0] = np.round(sol['F'],2)
            output_PSNR[ii,rreepp,0] =psnr  #peak signal-to-noise ratio
            
            "ARBPG with blocks:"
            idx_output = 0
            for block_size in block_sizes:
                idx_output += 1
                
                gap = 2*int(np.ceil(r/block_size)) #
                "ARBPG blocks"
                sol =     ARBPG_block(U0, V0, inst, tau_min, tau_max, sigma, block_size=block_size, gap=gap, prec=prec, stop_rule=0, max_stop=max_stop)
                print('\n ARBPG Blocks, size:',block_size)
                print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
                print("epochs", np.round(sol['epochs'],1))
                print('Fval=',sol['F'])
        
                
                UN = sol['U']
                VN = sol['V']
                
                IRN = UN.T@VN
                
                
                IRN_im_0 = (255.0 * IRN).astype(np.uint8)
                IRN_im = Image.fromarray(IRN_im_0)
                
                psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRN-IR,'fro')**2))
                print('PSNR=',psnr)
                
                output_epochs[ii,rreepp,idx_output] = np.round(sol['epochs'],1)
                output_time[ii,rreepp,idx_output] = np.round(sol['time'],2)
                output_Fval[ii,rreepp,idx_output] = np.round(sol['F'],2)
                output_PSNR[ii,rreepp,idx_output] =psnr  #peak signal-to-noise ratio
                
            "PALM:"        
            solB = ipalm_nmf(IR,U0,V0, r, max_iter=max_stop, tol=prec, alpha=0, beta=0)
            print('\n  PALM:')
            print(np.round(solB['time'],2), 'seconds ',solB['it'], 'epochs')
            print('Fval=',solB['F'])
            print("f eval=",solB['f_eval'])
          
            UB = solB['U']
            UP = solB['V']
                   
            IRB = UB.T@UP
            
            IRB_im_0 = (255.0 * IRB).astype(np.uint8)
            IRB_im = Image.fromarray(IRB_im_0)
            
            psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRB-IR,'fro')**2))
            print('PSNR=',psnr)
            
            output_epochs[ii,rreepp,4] = solB['it']
            output_time[ii,rreepp,4] = np.round(solB['time'],2)
            output_Fval[ii,rreepp,4] = np.round(solB['F'],2)
            output_PSNR[ii,rreepp,4] = psnr  #peak signal-to-noise ratio
            
            "iPALM:"        
            solB = ipalm_nmf(IR,U0,V0, r, max_iter=max_stop, tol=prec, alpha=0.2, beta=0.2)
            print('\n  iPALM :')
            print(np.round(solB['time'],2), 'seconds ',solB['it'], 'epochs')
            print('Fval=',solB['F'])
            print("f eval=",solB['f_eval'])
          
            UB = solB['U']
            VB = solB['V']
                   
            IRB = UB.T@UP
            
            IRB_im_0 = (255.0 * IRB).astype(np.uint8)
            IRB_im = Image.fromarray(IRB_im_0)
            
            psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRB-IR,'fro')**2))
            print('PSNR=',psnr)
            
            output_epochs[ii,rreepp,5] = solB['it']
            output_time[ii,rreepp,5] = np.round(solB['time'],2)
            output_Fval[ii,rreepp,5] = np.round(solB['F'],2)
            output_PSNR[ii,rreepp,5] = psnr  #peak signal-to-noise ratio 
            

            
            
plot1 = True
if plot1:
    

    
    metrics = {
    "# of epochs": output_epochs,
    "Time (s)": output_time,
    "Function value": output_Fval,
    "PSNR": output_PSNR
    }
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    alg_names = ["ARBPG $b=1$", "ARBPG $b=5$", "ARBPG $b=10$", "ARBPG $b=20$",  "PALM", "iPALM"] #[f"Algoritmo {i+1}" for i in range(7)]
    num_algs = len(alg_names)
    image_labels = ["Atacama", "Santiago", "Valdivia", "Niebla"]
    x_indices = np.arange(len(image_labels))
    
    markers = ['o']*4 + ['X']*2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    jitter = np.linspace(-0.2, 0.2, num_algs)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axs = axs.ravel()  
    
    for idx, (metric_name, data) in enumerate(metrics.items()):
        ax = axs[idx]
        
        mean_val = np.mean(data, axis=1)  
        std_val = np.std(data, axis=1)    
        
        for alg in range(num_algs):
            ax.errorbar(
                x_indices + jitter[alg], 
                mean_val[:, alg], 
                std_val[:, alg], 
                fmt=markers[alg], 
                color=colors[alg],
                ecolor=colors[alg], 
                elinewidth=1.2, 
                capsize=3, 
                markersize=7,
                alpha=0.85
            )
            
        ax.set_ylabel(metric_name)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(image_labels)
        ax.tick_params(labelbottom=True)
        for x in range(len(images)-1):
            ax.axvline(x+0.5,color="gray", linestyle=":",alpha=0.8,linewidth=1)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    

    fig_leg, ax_leg = plt.subplots(figsize=(8, 0.8))
    ax_leg.axis('off')  
    
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], 
                   markersize=9, label=alg_names[i])
        for i in range(num_algs)
    ]
    
    ax_leg.legend(handles=legend_elements, loc='center', ncol=num_algs, frameon=True, facecolor='#f9f9f9')
    
    plt.show()
    
    
    

" Experiment 2: Full images"
experiment2 = True
if experiment2 == True:
    
    
    theSeed = 1
    
    print('Seed: ',theSeed)
    random.seed(theSeed)
    
    images = [ 'Atacama_color', 'Santiago_color', 'Valdivia_color', 'Niebla_color']
    
    rs = [100, 100, 100, 150]
    
    output_epochs = np.zeros([len(images),3,4]) # images, rgb, alg
    output_time = np.zeros([len(images),3,4])
    output_Fval = np.zeros([len(images),3,4])
    output_PSNR = np.zeros([len(images),3,4]) #peak signal-to-noise ratio
        


    for ii in range(len(images)):
        
        r = rs[ii] # rank of the compression
        
        sel_image = images[ii]
        fig_name = '../Figures/'+sel_image+'.jpg'
        Im = plt.imread(fig_name)
        IR = Im[:,:,0]
        IG = Im[:,:,1]
        IB = Im[:,:,2]
        
        IR = IR.astype(float)/255
        IB = IB.astype(float)/255
        IG = IG.astype(float)/255
        
                  
        m, n = IR.shape
        
        
        "Algorithmic parameters"
        
        # ARBPG parameters:
        tau_min = 10**(-8)
        tau_max = 10**8
        sigma= 10**(-4)
        block_size = 5

        # Stopping parameters
        prec = 1e-4
        stop_rule = 0 
        max_stop = 1500000


        " Red images"
        
        
        print('\n ### Image:', sel_image, 'Color:', 'Red')
        
        inst = {}

        inst['A'] = IR
        inst['m'] = m
        inst['n'] = n
        inst['r'] = r
        

        # Random initialization:
        U0 = np.random.random([r,m])
        V0 = np.random.random([r,n])
        
        # Initialization by scikit
        W0_sk, H0_sk = _initialize_nmf(IR.T, n_components=r, init='nndsvdar', random_state=42)
        V0 = W0_sk.T
        U0 = H0_sk.copy()

        

        
        "ARBPG:" 
        gap = 2*r
        sol = ARBPG(U0,V0,inst,tau_min,tau_max,sigma,gap,prec,stop_rule=0,max_stop=max_stop)
        print('\n ARBPG:')
        print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
        print("epochs", np.round(sol['epochs'],1))
        print('Fval=',sol['F'])
        print("f eval=",sol['f_eval'])
        
        UN = sol['U']
        VN = sol['V']
        
        IRN = UN.T@VN
        
        
        IRN_im_0 = (255.0 * IRN).astype(np.uint8)
        IRN_im = Image.fromarray(IRN_im_0)

        psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRN-IR,'fro')**2))
        print('PSNR=',psnr)
        
        output_epochs[ii,0,0] = np.round(sol['epochs'],1)
        output_time[ii,0,0] = np.round(sol['time'],2)
        output_Fval[ii,0,0] = np.round(sol['F'],2)
        output_PSNR[ii,0,0] =psnr  #peak signal-to-noise ratio
        
        
        "ARBPG b=10:" 
        block_size = 5
        gap = 2*int(np.ceil(r/block_size))
        solB =     ARBPG_block(U0, V0, inst, tau_min, tau_max, sigma, block_size=block_size, gap=gap, prec=prec, stop_rule=0, max_stop=max_stop)
        print('\n ARBPG Blocks, size:',block_size)
        print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
        print("epochs", np.round(solB['epochs'],1))
        print('Fval=',solB['F'])
        print("f eval=",solB['f_eval'])
        
        UB = solB['U']
        VB = solB['V']
        
        IRB = UB.T@VB
        
        
        IRB_im_0 = (255.0 * IRB).astype(np.uint8)
        IRB_im = Image.fromarray(IRB_im_0)

        psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRB-IR,'fro')**2))
        print('PSNR=',psnr)
        
        output_epochs[ii,0,1] = np.round(solB['epochs'],1)
        output_time[ii,0,1] = np.round(solB['time'],2)
        output_Fval[ii,0,1] = np.round(solB['F'],2)
        output_PSNR[ii,0,1] =psnr  #peak signal-to-noise ratio
        

                
        
        "iPALM"
        solP = ipalm_nmf(IR,U0,V0, r, max_iter=max_stop, tol=prec, alpha=0, beta=0)
        print('\n  iPALM 0.2:')
        print(np.round(solP['time'],2), 'seconds ',solP['it'], 'iterations')
        print('Fval=',solP['F'])
        print("f eval=",solP['f_eval'])

        
        UP = solP['U']
        VP = solP['V']
        
        
        IRP = UP.T@VP
        
        IRP_im_0 = (255.0 * IRP).astype(np.uint8)
        IRP_im = Image.fromarray(IRP_im_0)
        
        psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRP-IR,'fro')**2))
        print('PSNR=',psnr)

        
        output_epochs[ii,0,2] = solP['it']
        output_time[ii,0,2] = np.round(solP['time'],2)
        output_Fval[ii,0,2] = np.round(solP['F'],2)
        output_PSNR[ii,0,2] =psnr  #peak signal-to-noise ratio
        
        
        "scikit "
        tic_scikit = time()
        model = NMF(n_components=r,init="nndsvda",tol=prec,solver="cd",random_state=42,max_iter = int(np.floor(solB['epochs'])))
        W = model.fit_transform(IR.T)
        H = model.components_
        UM = H
        VM = W.T 
        FvalS = .5*norm(IR-UM.T@VM)**2
        time_S = time()-tic_scikit
                
        print('\n Scikit-learn-NMF:')
        print(np.round(time_S,2), 'seconds ',model.n_iter_, 'iterations')
        print('Fval=',FvalS)
        
        IRS = UM.T@VM
        
        IRS_im_0 = (255.0 * IRS).astype(np.uint8)
        IRS_im = Image.fromarray(IRS_im_0)
        
        psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRS-IR,'fro')**2))
        print('PSNR=',psnr)

        
        output_epochs[ii,0,3] = model.n_iter_
        output_time[ii,0,3] = np.round(time_S,2)
        output_Fval[ii,0,3] = np.round(FvalS,2)
        output_PSNR[ii,0,3] =psnr  #peak signal-to-noise ratio


        
        " Green images"
        print('\n ### Image:', sel_image, 'Color:', 'Green')

        inst = {}
        
        inst['A'] = IG
        inst['m'] = m
        inst['n'] = n
        inst['r'] = r

        # Random initialization
        # U0 = np.random.random([r,m])
        # V0 = np.random.random([r,n])
        # Sci-kit initialization
        W0_sk, H0_sk = _initialize_nmf(IG.T, n_components=r, init='nndsvdar', random_state=42)
        V0 = W0_sk.T
        U0 = H0_sk.copy()
        

        "ARBPG" 
        gap = 2*r
        sol = ARBPG(U0,V0,inst,tau_min,tau_max,sigma,gap,prec,stop_rule=0,max_stop=max_stop)
        print('\n ARBPG:')
        print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
        print("epochs", np.round(sol['epochs'],1))
        print('Fval=',sol['F'])
        print("f eval=",sol['f_eval'])

        
        UN = sol['U']
        VN = sol['V']
        
        IGN = UN.T@VN
        
        
        IGN_im_0 = (255.0 * IGN).astype(np.uint8)
        IGN_im = Image.fromarray(IGN_im_0)
        
        psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGN-IG,'fro')**2))
        print('PSNR=',psnr)
        
        output_epochs[ii,1,0] = np.round(sol['epochs'],1)
        output_time[ii,1,0] = np.round(sol['time'],2)
        output_Fval[ii,1,0] = np.round(sol['F'],2)
        output_PSNR[ii,1,0] =psnr  #peak signal-to-noise ratio
        
        
        "ARBPG b=10:" 
        block_size = 5
        gap = 2*int(np.ceil(r/block_size))
        solB =     ARBPG_block(U0, V0, inst, tau_min, tau_max, sigma, block_size=block_size, gap=gap, prec=prec, stop_rule=0, max_stop=max_stop)
        print('\n ARBPG Blocks, size:',block_size)
        print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
        print("epochs", np.round(solB['epochs'],1))
        print('Fval=',solB['F'])
        print("f eval=",solB['f_eval'])
        
        UB = solB['U']
        VB = solB['V']
        
        IGB = UB.T@VB
        
        
        IGB_im_0 = (255.0 * IGB).astype(np.uint8)
        IGB_im = Image.fromarray(IGB_im_0)

        psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGB-IG,'fro')**2))
        print('PSNR=',psnr)
        
        output_epochs[ii,1,1] = np.round(solB['epochs'],1)
        output_time[ii,1,1] = np.round(solB['time'],2)
        output_Fval[ii,1,1] = np.round(solB['F'],2)
        output_PSNR[ii,1,1] =psnr  #peak signal-to-noise ratio
        
        
        "iPALM"
        solP = ipalm_nmf(IG,U0,V0, r, max_iter=max_stop, tol=prec, alpha=0, beta=0)
        print('\n  iPALM 0.2:')
        print(np.round(solP['time'],2), 'seconds ',solP['it'], 'iterations')
        print('Fval=',solP['F'])
        print("f eval=",solP['f_eval'])

        
        UP = solP['U']
        VP = solP['V']
        
        
        IGP = UP.T@VP
        
        IGP_im_0 = (255.0 * IGP).astype(np.uint8)
        IGP_im = Image.fromarray(IGP_im_0)
        
        psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGP-IG,'fro')**2))
        print('PSNR=',psnr)

        
        output_epochs[ii,1,2] = solP['it']
        output_time[ii,1,2] = np.round(solP['time'],2)
        output_Fval[ii,1,2] = np.round(solP['F'],2)
        output_PSNR[ii,1,2] =psnr  #peak signal-to-noise ratio
        
        
        "scikit "
        tic_scikit = time()
        model = NMF(n_components=r,init="nndsvda",tol=prec,solver="cd",random_state=42,max_iter = int(np.floor(solB['epochs'])))
        W = model.fit_transform(IG.T)
        H = model.components_
        UM = H
        VM = W.T 
        FvalS = .5*norm(IG-UM.T@VM)**2
        time_S = time()-tic_scikit
                
        print('\n Scikit-learn-NMF:')
        print(np.round(time_S,2), 'seconds ',model.n_iter_, 'iterations')
        print('Fval=',FvalS)
        
        IGS = UM.T@VM
        
        IGS_im_0 = (255.0 * IGS).astype(np.uint8)
        IGS_im = Image.fromarray(IGS_im_0)
        
        psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGS-IG,'fro')**2))
        print('PSNR=',psnr)

        
        output_epochs[ii,1,3] = model.n_iter_
        output_time[ii,1,3] = np.round(time_S,2)
        output_Fval[ii,1,3] = np.round(FvalS,2)
        output_PSNR[ii,1,3] =psnr  #peak signal-to-noise ratio

        
        

        
        
                
        
        " Blue images"
        print('\n ### Image:', sel_image, 'Color:', 'Blue')

        inst = {}
        
        inst['A'] = IB
        inst['m'] = m
        inst['n'] = n
        inst['r'] = r

        # Random initialization
        # U0 = np.random.random([r,m])
        # V0 = np.random.random([r,n])
        
        # Scikit-initialization
        W0_sk, H0_sk = _initialize_nmf(IB.T, n_components=r, init='nndsvdar', random_state=42)
        V0 = W0_sk.T
        U0 = H0_sk.copy()


        "ARBPG:" 
        gap = 2*r
        sol = ARBPG(U0,V0,inst,tau_min,tau_max,sigma,gap,prec,stop_rule=0,max_stop=max_stop)
        print('\n ARBPG:')
        print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
        print("epochs", np.round(sol['epochs'],1))
        print('Fval=',sol['F'])
        print("f eval=",sol['f_eval'])

        
        UN = sol['U']
        VN = sol['V']
        
        IBN = UN.T@VN
        
        
        IBN_im_0 = (255.0 * IBN).astype(np.uint8)
        IBN_im = Image.fromarray(IBN_im_0)
        
        psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBN-IB,'fro')**2))
        print('PSNR=',psnr)
        
        output_epochs[ii,2,0] = np.round(sol['epochs'],1)
        output_time[ii,2,0] = np.round(sol['time'],2)
        output_Fval[ii,2,0] = np.round(sol['F'],2)
        output_PSNR[ii,2,0] =psnr  #peak signal-to-noise ratio
        
        "ARBPG b=10:" 
        block_size = 5
        gap = 2*int(np.ceil(r/block_size))
        solB =     ARBPG_block(U0, V0, inst, tau_min, tau_max, sigma, block_size=block_size, gap=gap, prec=prec, stop_rule=0, max_stop=max_stop)
        print('\n ARBPG Blocks, size:',block_size)
        print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
        print("epochs", np.round(solB['epochs'],1))
        print('Fval=',solB['F'])
        print("f eval=",solB['f_eval'])
        
        UB = solB['U']
        VB = solB['V']
        
        IBB = UB.T@VB
        
        
        IBB_im_0 = (255.0 * IBB).astype(np.uint8)
        IBB_im = Image.fromarray(IBB_im_0)

        psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBB-IB,'fro')**2))
        print('PSNR=',psnr)
        
        output_epochs[ii,2,1] = np.round(solB['epochs'],1)
        output_time[ii,2,1] = np.round(solB['time'],2)
        output_Fval[ii,2,1] = np.round(solB['F'],2)
        output_PSNR[ii,2,1] =psnr  #peak signal-to-noise ratio
        
        "iPALM"
        solP = ipalm_nmf(IB,U0,V0, r, max_iter=max_stop, tol=prec, alpha=0, beta=0)
        print('\n  iPALM 0.2:')
        print(np.round(solP['time'],2), 'seconds ',solP['it'], 'iterations')
        print('Fval=',solP['F'])
        print("f eval=",solP['f_eval'])

        
        UP = solP['U']
        VP = solP['V']
        
        
        IBP = UP.T@VP
        
        IBP_im_0 = (255.0 * IBP).astype(np.uint8)
        IBP_im = Image.fromarray(IBP_im_0)
        
        psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBP-IB,'fro')**2))
        print('PSNR=',psnr)

        
        output_epochs[ii,2,2] = solP['it']
        output_time[ii,2,2] = np.round(solP['time'],2)
        output_Fval[ii,2,2] = np.round(solP['F'],2)
        output_PSNR[ii,2,2] =psnr  #peak signal-to-noise ratio


        "scikit "
        tic_scikit = time()
        model = NMF(n_components=r,init="nndsvda",tol=prec,solver="cd",random_state=42,max_iter = int(np.floor(solB['epochs'])))
        W = model.fit_transform(IB.T)
        H = model.components_
        UM = H
        VM = W.T 
        FvalS = .5*norm(IB-UM.T@VM)**2
        time_S = time()-tic_scikit
                
        print('\n Scikit-learn-NMF:')
        print(np.round(time_S,2), 'seconds ',model.n_iter_, 'iterations')
        print('Fval=',FvalS)
        
        IBS = UM.T@VM
        
        IBS_im_0 = (255.0 * IBS).astype(np.uint8)
        IBS_im = Image.fromarray(IBS_im_0)
        
        psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBS-IB,'fro')**2))
        print('PSNR=',psnr)

        
        output_epochs[ii,2,3] = model.n_iter_
        output_time[ii,2,3] = np.round(time_S,2)
        output_Fval[ii,2,3] = np.round(FvalS,2)
        output_PSNR[ii,2,3] =psnr  #peak signal-to-noise ratio

        
        
        
        
        " Saving the images: "
        
        IN_im = np.concatenate((IRN_im_0[...,None],IGN_im_0[...,None],IBN_im_0[...,None]),axis=2)
        IN_im = Image.fromarray(IN_im)
        IN_im.save(sel_image+'_ARBPG.jpg')
        
        IBo_im = np.concatenate((IRB_im_0[...,None],IGB_im_0[...,None],IBB_im_0[...,None]),axis=2)
        IBo_im = Image.fromarray(IBo_im)
        IBo_im.save(sel_image+'_ARBPG_Blocks.jpg')
        
        IM_im = np.concatenate((IRP_im_0[...,None],IGP_im_0[...,None],IBP_im_0[...,None]),axis=2)
        IM_im = Image.fromarray(IM_im)
        IM_im.save(sel_image+'_iPALM.jpg')
        
        IS_im = np.concatenate((IRS_im_0[...,None],IGS_im_0[...,None],IBS_im_0[...,None]),axis=2)
        IS_im = Image.fromarray(IS_im)
        IS_im.save(sel_image+'_SCIKIT.jpg')
        
        np.savez('NMF_ex2', output_epochs, output_time, 
                 output_Fval, output_PSNR) 
        



    "Compute average value over colors for the  table"

    output_epochs_mean = np.round(np.mean(output_epochs, axis=1),1)
    output_time_mean = np.round(np.mean(output_time,axis=1),2)
    output_Fval_mean = np.round(np.mean(output_Fval,axis=1),2)
    output_PSNR_mean = np.round(np.mean(output_PSNR,axis=1),2)



    fname = "Table_NMF_ex2"
    DRNMF.generate_table_epochs(fname, images, output_epochs_mean, output_time_mean, output_Fval_mean, output_PSNR_mean)


