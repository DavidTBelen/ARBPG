# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:29:11 2025

Author: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper: Randomized block proximal method with locally
Lipschitz continuous gradients
"""



import numpy as np
from numpy import random
from Algorithms import NMF_adaptive_R1 as NMF
import matplotlib.pyplot as plt
from Auxiliaries import DisplayResultsNMF as DRNMF





#theSeed = np.random.choice(20000,1)[0]
theSeed = 1

print('Seed: ',theSeed)
random.seed(theSeed)

images = ['Atacama_very_small']

rep = 5

# Saving output monotone method
output_iterations_M = np.zeros([3,rep]) # 3 strategies
output_time_M = np.zeros([3,rep])
output_Fval_M = np.zeros([3,rep])
output_feval_M = np.zeros([3,rep])

# Saving output nonmonotone method
output_iterations_N = np.zeros([3,rep]) # 3 strategies
output_time_N = np.zeros([3,rep])
output_Fval_N = np.zeros([3,rep])
output_feval_N = np.zeros([3,rep])


# Saving output Boosted method
output_iterations_B = np.zeros([6,rep]) # 3 strategies
output_time_B = np.zeros([6,rep])
output_Fval_B = np.zeros([6,rep])
output_feval_B = np.zeros([6,rep])



rs = [35]

for ii in range(len(images)):
    
    r = rs[ii] # rank of the compression
    
    sel_image = images[ii]
    
    print('\n ####### IMAGE: ',  sel_image, ' #######')
    fig_name = 'Figures/'+sel_image+'.jpg'
    Im = plt.imread(fig_name)
    IR = Im[:,:,0]
    IB = Im[:,:,1]
    IG = Im[:,:,2]
    
    IR = IR.astype(float)/255
    IB = IB.astype(float)/255
    IG = IG.astype(float)/255
    
    
    
    
    m, n = IR.shape
    
    
    "Algorithmic parameters"
    
    # Boosted RNBPG parameters:
    M = 11
    beta = .9
    tau_min = 10**(-8)
    tau_max = 10**8
    sigma= 10**(-4)
    
    # Boosted RNBPG parameters:
    blam0 = 3
    rho = .5
    alpha= .1
    
    # Stopping parameters
    gap = 2*(m+n)
    prec = 1e-4
    stop_rule = 0 
    max_stop = 200000
    
    btau = 2
    
    inst = {}
    
    inst['A'] = IR
    inst['m'] = m
    inst['n'] = n
    inst['r'] = r
    
    for rr in range(rep):
        # Always same initial point:
        U0 = np.random.random([r,m])
        V0 = np.random.random([r,n])
            
        print('\n ### Fixed proximal stepsizes:')
        
        print('\n Monotone:')
        sol = NMF.RNBPG_btau_fixed(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
        print('Fval=',sol['F'])
        print('f eval.=',sol['f_eval'])

        
        print('\n Nonmonotone:')
        solN = NMF.RNBPG_btau_fixed(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solN['time'],2), 'seconds ',solN['it'], 'iterations')
        print('Fval=',solN['F'])
        print('f eval.=',solN['f_eval'])

        
        print('\n Boosted:')
        solB = NMF.BRNBPG_btau_fixed_nonadapt(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
        print('Fval=',solB['F'])
        print('f eval.=',solB['f_eval'])

        
        print('\n Boosted Adaptive:')
        solBA = NMF.BRNBPG_btau_fixed(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solBA['time'],2), 'seconds ',solBA['it'], 'iterations')
        print('Fval=',solBA['F'])
        print('f eval.=',solBA['f_eval'])

        
        # Saving output monotone method
        output_iterations_M[0,rr] =  sol['it']
        output_time_M[0,rr] = sol['time'] # 
        output_Fval_M[0,rr] = sol['F']
        output_feval_M[0,rr] = sol['f_eval']

        # Saving output nonmonotone method
        output_iterations_N[0,rr] = solN['it'] 
        output_time_N[0,rr] = solN['time']
        output_Fval_N[0,rr] = solN['F']
        output_feval_N[0,rr] = solN['f_eval']


        # Saving output Boosted method
        output_iterations_B[0,rr] = solB['it'] 
        output_time_B[0,rr] = solB['time']
        output_Fval_B[0,rr] = solB['F']
        output_feval_B[0,rr] = solB['f_eval']

        
        output_iterations_B[1,rr] = solBA['it'] 
        output_time_B[1,rr] = solBA['time']
        output_Fval_B[1,rr] = solBA['F']
        output_feval_B[1,rr] = solBA['f_eval']


        
        
        
        
        print('\n ### Decreasing proximal stepsizes:')
        
        print('\n Monotone:')
        sol = NMF.RNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
        print('Fval=',sol['F'])
        print('f eval.=',sol['f_eval'])
        
        print('\n Nonmonotone:')
        solN = NMF.RNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solN['time'],2), 'seconds ',solN['it'], 'iterations')
        print('Fval=',solN['F'])
        print('f eval.=',solN['f_eval'])

        
        print('\n Boosted:')
        solB = NMF.BRNBPG_btau_dec_nonadapt(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
        print('Fval=',solB['F'])
        print('f eval.=',solB['f_eval'])

        
        print('\n Boosted Adaptive:')
        solBA = NMF.BRNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solBA['time'],2), 'seconds ',solBA['it'], 'iterations')
        print('Fval=',solBA['F'])
        print('f eval.=',solBA['f_eval'])

        
        # Saving output monotone method
        output_iterations_M[1,rr] =  sol['it']
        output_time_M[1,rr] = sol['time'] # 
        output_Fval_M[1,rr] = sol['F']
        output_feval_M[1,rr] = sol['f_eval']


        # Saving output nonmonotone method
        output_iterations_N[1,rr] = solN['it'] 
        output_time_N[1,rr] = solN['time']
        output_Fval_N[1,rr] = solN['F']
        output_feval_N[1,rr] = solN['f_eval']


        # Saving output Boosted method
        output_iterations_B[2,rr] = solB['it'] 
        output_time_B[2,rr] = solB['time']
        output_Fval_B[2,rr] = solB['F']
        output_feval_B[2,rr] = solB['f_eval']

        
        output_iterations_B[3,rr] = solBA['it'] 
        output_time_B[3,rr] = solBA['time']
        output_Fval_B[3,rr] = solBA['F']
        output_feval_B[3,rr] = solBA['f_eval']

        
        
        print('\n ### Self-Adaptive proximal stepsizes:')
        
        print('\n Monotone:')
        sol = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
        print('Fval=',sol['F'])
        print('f eval.=',sol['f_eval'])

        
        print('\n Nonmonotone:')
        solN = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solN['time'],2), 'seconds ',solN['it'], 'iterations')
        print('Fval=',solN['F'])
        print('f eval.=',solN['f_eval'])

        
        print('\n Boosted Adaptive on btau:')
        solB = NMF.BRNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
        print('Fval=',solB['F'])
        print('f eval.=',solB['f_eval'])

        
        
        print('\n Boosted Double Adaptive:')
        solBA = NMF.BRNBPG_btau_double_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
        print(np.round(solBA['time'],2), 'seconds ',solBA['it'], 'iterations')
        print('Fval=',solBA['F'])
        print('f eval.=',solBA['f_eval'])

        

        
        
        
        # Saving output monotone method
        output_iterations_M[2,rr] =  sol['it']
        output_time_M[2,rr] = sol['time'] # 
        output_Fval_M[2,rr] = sol['F']
        output_feval_M[2,rr] = sol['f_eval']


        #Saving output nonmonotone method
        output_iterations_N[2,rr] = solN['it'] 
        output_time_N[2,rr] = solN['time']
        output_Fval_N[2,rr] = solN['F']
        output_feval_N[2,rr] = solN['f_eval']


        # Saving output Boosted method
        output_iterations_B[4,rr] = solB['it']
        output_time_B[4,rr] = solB['time']
        output_Fval_B[4,rr] = solB['F']
        output_feval_B[4,rr] = solB['f_eval']

        
        output_iterations_B[5,rr] = solBA['it'] 
        output_time_B[5,rr] = solBA['time']
        output_Fval_B[5,rr] = solBA['F']
        output_feval_B[5,rr] = solBA['f_eval']

        
        # np.savez('Results/Results_ExNMF_adaptive_R1', output_iterations_M, 
        #           output_iterations_N, output_iterations_B, output_time_M,
        #           output_time_N, output_time_B, output_Fval_M, output_Fval_N,
        #           output_Fval_B,output_feval_M,output_feval_N,
        #           output_feval_B)
        
        
"Compute average value over repetitions"

means_iterations_M = np.mean(output_iterations_M, axis=1)
means_iterations_N = np.mean(output_iterations_N, axis=1)
means_iterations_B = np.mean(output_iterations_B, axis=1)

means_time_M = np.round(np.mean(output_time_M, axis=1),2)
means_time_N = np.round(np.mean(output_time_N, axis=1),2)
means_time_B = np.round(np.mean(output_time_B, axis=1),2)


means_Fval_M = np.round(np.mean(output_Fval_M, axis=1),2)
means_Fval_N = np.round(np.mean(output_Fval_N, axis=1),2)
means_Fval_B = np.round(np.mean(output_Fval_B, axis=1),2)

means_feval_M = np.round(np.mean(output_feval_M, axis=1),2)
means_feval_N = np.round(np.mean(output_feval_N, axis=1),2)
means_feval_B = np.round(np.mean(output_feval_B, axis=1),2)


fname1 =   "Results/Table_ExNMF_strategy_M_R1"
DRNMF.table_strategies_MN(fname1, means_iterations_M, means_time_M, means_Fval_M,
                          means_feval_M)
        
fname2 =   "Results/Table_ExNMF_strategy_N_R1"
DRNMF.table_strategies_MN(fname2, means_iterations_N, means_time_N, means_Fval_N,
                          means_feval_N)
        
fname2 =   "Results/Table_ExNMF_strategy_B_R1"
DRNMF.table_strategies_B(fname2, means_iterations_B, means_time_B, means_Fval_B,
                         means_feval_B)
        
        
        
