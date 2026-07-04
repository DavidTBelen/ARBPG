# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:05:11 2026

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper: Randomized block proximal method with locally
Lipschitz continuous gradient
"""


import numpy as np
from numpy import random
from Auxiliaries import DisplayResultsNMF as DRNMF
from time import time
from numpy.linalg import norm

" Algorithms "

"ARBPG"
def ARBPG(U0, inst, tau_min, tau_max, sigma, beta, gap=50, prec=1e-6, stop_rule=0, max_stop=1000):
    A = inst['A']
    r = inst['r']
    
    normA = norm(A)
    
    U = U0.copy()
    
    f_eval = 0
    AU = A - U.T @ U
    
    F_old = 0.5 * norm(AU, 'fro')**2
    F_list = [F_old]  
    
    counter = [0, 0]
    t0 = time()
    
    continua = True
    it = 0
    
    while continua:
        ik = random.randint(0, r)
        # ---------------------------------------------------------------------
        # Update U
        # ---------------------------------------------------------------------
        ui = U[ik, :].copy()
        ui_norm_sq = np.dot(ui,ui)

        

        tau =  1/(2*np.sqrt(2*F_old) +  4*np.dot(ui,ui)) 
        tau = max(min(tau_max, tau), tau_min)
        
        gf = -2*ui @ AU.T
        
        adaptive_ls = True
        while adaptive_ls:
            temp_ui = ui - tau * gf
            di = -ui + np.maximum(temp_ui, 0.)
            di_norm_sq = np.dot(di, di)
            uidi = np.dot(di,ui)
            
            if di_norm_sq > 1e-14:
                
                linear_term = np.dot(gf,di) - np.dot(di,np.dot(AU,di))
                quad_term = ui_norm_sq*di_norm_sq + 0.5*di_norm_sq**2 + uidi**2 + 2*uidi*di_norm_sq
                temp_arg = quad_term + linear_term

                f_eval += 1
                
                if temp_arg <= -sigma * di_norm_sq:
                    U[ik, :] += di
                    AU -=  (np.outer(di, ui)+np.outer(ui,di)+ np.outer(di,di))
                    norm(AU-(A-U.T@U))
                    
                    F_new = F_old + temp_arg
                    F_list.append(F_new)
                    F_old = F_new
                    adaptive_ls = False
                else:
                    tau = beta * tau
            else:
                F_list.append(F_old)
                break
        

         
        # ---------------------------------------------------------------------
        # Check stopping condition
        # ---------------------------------------------------------------------
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        # O(1) indexing from Python list
        error_crit = abs(F_list[-min(gap, it + 1)] - F_old) /  normA
        
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
        'F': F_old,
        'F_list': np.array(F_list),  
        'it': counter[0],
        'time': counter[1],
        'f_eval': f_eval,
    }
    
    return solution


"Boosted ARBPG"
def boosted_ARBPG(U0, inst, tau_min, tau_max, sigma, beta, blam0, rho, alpha, gap=50, prec=1e-6, stop_rule=0, max_stop=1000):
    A = inst['A']
    r = inst['r']
    
    normA = norm(A)
    
    U = U0.copy()
    
    f_eval = 0
    AU = A - U.T @ U
    
    F_old = 0.5 * norm(AU, 'fro')**2
    F_list = [F_old]  
    lam_list = []
    
    blam = blam0
    boosted_accepted = 0
    
    counter = [0, 0]
    t0 = time()
    
    continua = True
    it = 0
    
    while continua:
        ik = random.randint(0, r)
        # ---------------------------------------------------------------------
        # Update U
        # ---------------------------------------------------------------------
        ui = U[ik, :].copy()
        ui_norm_sq = np.dot(ui,ui)

        
        tau = 1/(2*np.sqrt(2*F_old) +  4*np.dot(ui,ui))
        tau = max(min(tau_max, tau), tau_min)
        
        gf = -2*ui @ AU.T
        
        adaptive_ls = True
        while adaptive_ls:
            temp_ui = ui - tau * gf
            di = -ui + np.maximum(temp_ui, 0.)
            di_norm_sq = np.dot(di, di)
            uidi = np.dot(di,ui)
            
            if di_norm_sq > 1e-14:
                linear_term = np.dot(gf,di) - np.dot(di,np.dot(AU,di))
                quad_term = ui_norm_sq*di_norm_sq + 0.5*di_norm_sq**2 + uidi**2 + 2*uidi*di_norm_sq
                temp_arg = quad_term + linear_term
                
                f_eval += 1
                
                if temp_arg <= -sigma * di_norm_sq:
                    U[ik, :] += di
                    AU -=  (np.outer(di, ui)+np.outer(ui,di)+ np.outer(di,di))
                    norm(AU-(A-U.T@U))
                    
                    F_new = F_old + temp_arg
                    F_list.append(F_new)
                    F_old = F_new
                    adaptive_ls = False
                    
                    lam = blam
                    ui = U[ik,:].copy()
                    gf = -2*ui @ AU.T 
                    counter_b = 0 
                    boosted_ls = True
                    while boosted_ls: 
                        deli = (lam-1)*di 
                        index_hat = (U[ik,:]+deli)<0 
                        if sum(index_hat)>0: 
                            lam = rho*lam 
                            counter_b += 1 
                        else: 
                            deli_norm_sq = np.dot(deli,deli) 
                            ui_norm_sq = np.dot(ui, ui)
                            uideli = np.dot(deli,ui)
                            
                            linear_term = np.dot(gf,deli) - np.dot(deli,np.dot(AU,deli))
                            quad_term =  ui_norm_sq*deli_norm_sq + 0.5*deli_norm_sq**2 + uideli**2 + 2*uideli*deli_norm_sq
                            temp_arg = quad_term + linear_term
                            
                            f_eval += 1 
                            if lam <= 1 or temp_arg <= -alpha * deli_norm_sq:
                                boosted_ls = False 
                            else: 
                                lam = rho*lam 
                                counter_b += 1 
                    
                    lam_list.append(lam) 
                    if lam > 1: 
                        boosted_accepted += 1
                        
                        U[ik,:] += deli 
                        AU -= (np.outer(deli, ui)+np.outer(ui,deli)+ np.outer(deli,deli))
                        
                        F_new = F_old + temp_arg
                        F_list.append(F_new)
                        F_old = F_new
                        
                        if counter_b == 0:
                            blam = 2*blam
                        else:
                            blam = max(blam0,lam)
                    else:
                        blam = blam0
                else:
                    tau = beta * tau
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


    lam_list = np.array(lam_list) -1.
    solution = {
        'U': U,
        'F': F_old,
        'F_list': np.array(F_list),  
        'it': counter[0],
        'time': counter[1],
        'f_eval': f_eval,
        'n_boosted': boosted_accepted,
        'lam_list': lam_list,
        'lam_avg': np.mean(lam_list),
        'lam_max': np.max(lam_list),
    }
    
    return solution


# Start of the experiment:
    
saveT = True
theSeed =  1 
print('Seed: ',theSeed)
random.seed(theSeed)

# Setting of the experiment
numbers_samples = [500,800,1000] 
rep = 10

output_iterations = np.zeros([len(numbers_samples),rep,2])
output_time = np.zeros([len(numbers_samples),rep,2])
output_Fval = np.zeros([len(numbers_samples),rep,2])
output_nfeval = np.zeros([len(numbers_samples),rep,2])
output_accuracy = np.zeros([len(numbers_samples),rep,2])
output_ARI = np.zeros([len(numbers_samples),rep,2])
output_nbls = np.zeros([len(numbers_samples),rep]) #number of boosted ls
output_lamavg = np.zeros([len(numbers_samples),rep]) 
output_lammax = np.zeros([len(numbers_samples),rep]) 


experiment3 = True
if experiment3:
    from sklearn.datasets import load_digits
    from sklearn.neighbors import kneighbors_graph
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import accuracy_score, adjusted_rand_score
    
    def evaluate_symnmf_clustering(U, true_labels):
        
        predicted_clusters = np.argmax(U, axis=0)
        
        n_clusters = U.shape[0]
        n_classes = len(np.unique(true_labels))
        
        cost_matrix = np.zeros((n_clusters, n_classes))
        for i in range(n_clusters):
            for j in range(n_classes):
                cost_matrix[i, j] = -np.sum((predicted_clusters == i) & (true_labels == j))
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapped_predictions = np.zeros_like(predicted_clusters)
        for i, j in zip(row_ind, col_ind):
            mapped_predictions[predicted_clusters == i] = j
            
        acc = accuracy_score(true_labels, mapped_predictions)
        ari = adjusted_rand_score(true_labels, predicted_clusters) # ARI is invariant to label names
        
        print("="*40)
        print("      SymNMF CLUSTERING PERFORMANCE      ")
        print("="*40)
        print(f"Clustering Accuracy (ACC): {acc * 100:.2f}%")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}  (1.0 is perfect)")
        print("="*40)
        
        return acc, ari, mapped_predictions
    
    for nn in range(len(numbers_samples)):
        
        
        for rreepp in range(rep):
            
            print("\n Number of samples considered: ",numbers_samples[nn], "Repetition", rreepp)
            X, y = load_digits(return_X_y=True)
            
            
            n_samples = numbers_samples[nn]
            
            idx = np.random.choice(range(len(y)), size =n_samples )
            X_sub = X[idx]
            true_labels = y[idx]
            
            
            
            K = int(np.ceil(np.log(n_samples)))
            A_sparse = kneighbors_graph(X_sub, n_neighbors=K, mode='connectivity', include_self=False)
            A = A_sparse.toarray()
            A = 0.5 * (A + A.T)
                                        
            np.fill_diagonal(A, 0.0)
            n = n_samples
                     
            r = 10  
            
            # ARBPG parameters:
            beta = .9
            tau_min = 10**(-8)
            tau_max = 10**8
            sigma= 10**(-4)
        
            # Stopping parameters
            gap = 2*r
            prec = 1e-4
            stop_rule = 0 
            max_stop = 100000
        
            U0 = np.random.random([r,n])
             
            seed_algorithms = random.randint(0,10000)
        
            inst = {}
        
            inst['A'] = A
            inst['n'] = n
            inst['r'] = r
                
            "ARBPG:" 
            random.seed(seed_algorithms)
            sol = ARBPG(U0,inst,tau_min,tau_max,sigma,beta,gap,prec,stop_rule=stop_rule,max_stop=max_stop)
            print('\n ARBPG:')
            print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
            print('Fval=',sol['F'])
            print("f eval=",sol['f_eval'])
            acc, ari, mapped_predictions = evaluate_symnmf_clustering(sol['U'], true_labels)
        
            U = sol['U']
            
            # Store values
            output_iterations[nn,rreepp,0] = int(np.round(sol['it']))
            output_time[nn,rreepp,0] = np.round(sol['time'],2)
            output_Fval[nn,rreepp,0] = np.round(sol['F'],2)
            output_nfeval[nn,rreepp,0] = np.round(sol['f_eval'],2)
            output_accuracy[nn,rreepp,0] = acc
            output_ARI[nn,rreepp,0] = ari
        
        
            "Boosted ARBPG:" 
        
            # Boosted parameters:
            blam0 = 2
            rho = 0.5
            alpha = 0.1
        
            random.seed(seed_algorithms)
            solB = boosted_ARBPG(U0, inst, tau_min, tau_max, sigma, beta, blam0, rho, alpha, gap=gap, prec=1e-16, stop_rule=1, max_stop=sol['time'])
            print('\n Boosted ARBPG:')
            print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
            print('Fval=',solB['F'])
            print("f eval=",solB['f_eval'])
            print("# boosted accepted=", solB["n_boosted"])
            accB, ariB, mapped_predictionsB = evaluate_symnmf_clustering(solB['U'], true_labels)
        
            # Store values
            output_iterations[nn,rreepp,1] = int(np.round(solB['it']))
            output_time[nn,rreepp,1] = np.round(solB['time'],2)
            output_Fval[nn,rreepp,1] = np.round(solB['F'],2)
            output_nfeval[nn,rreepp,1] = np.round(solB['f_eval'],2)
            output_accuracy[nn,rreepp,1] = accB
            output_ARI[nn,rreepp,1] = ariB
            output_nbls[nn,rreepp] = solB["n_boosted"]
            output_lamavg[nn,rreepp] = solB["lam_avg"]
            output_lammax[nn,rreepp] = solB["lam_max"]

       


    
    output_iterations_mean = np.round(np.mean(output_iterations, axis=1),0)
    output_time_mean = np.round(np.mean(output_time,axis=1),2)
    output_Fval_mean = np.round(np.mean(output_Fval,axis=1),2)
    output_nfeval_mean = np.round(np.mean(output_nfeval,axis=1),2)
    output_accuracy_mean = np.round(np.mean(output_accuracy,axis=1),4)*100
    output_ARI_mean = np.round(np.mean(output_ARI,axis=1),2)
    output_nbls_mean = np.round(np.mean(output_nbls,axis=1),2)
    out_lamavg_mean = np.round(np.mean(output_lamavg,axis=1),4)
    out_lammax_mean = np.round(np.mean(output_lammax,axis=1),4)

    
    output_iterations_sdv = np.round(np.std(output_iterations, axis=1),0)
    output_time_sdv = np.round(np.std(output_time,axis=1),2)
    output_Fval_sdv = np.round(np.std(output_Fval,axis=1),2)
    output_nfeval_sdv = np.round(np.std(output_nfeval,axis=1),2)
    output_accuracy_sdv = np.round(np.std(output_accuracy,axis=1),4)*100
    output_ARI_sdv = np.round(np.std(output_ARI,axis=1),2)
    output_nbls_sdv = np.round(np.std(output_nbls,axis=1),2)
    out_lamavg_sdv = np.round(np.std(output_lamavg,axis=1),4)
    out_lammax_sdv = np.round(np.std(output_lammax,axis=1),4)




    if saveT == True:
       
        fname = "Table_SymNMF"
        DRNMF.generate_table_SymNMF_sdv(fname, numbers_samples, output_iterations_mean, output_nfeval_mean, 
                                    output_accuracy_mean, output_ARI_mean, output_nbls_mean, 
                                    output_iterations_sdv, output_nfeval_sdv, 
                                    output_accuracy_sdv, output_ARI_sdv, output_nbls_sdv)
        
        fname= "Table_SymNMF_boosted"
        DRNMF.generate_table_boosted(fname, numbers_samples, out_lamavg_mean, out_lammax_mean, out_lamavg_sdv, out_lammax_sdv)
