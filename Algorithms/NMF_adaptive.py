# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:27:16 2025

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper:  Randomized block proximal method with locally
Lipschitz continuous gradient
"""

import numpy as np
from time import time
from numpy import random
from numpy.linalg import norm



" Auxiliary functions:"


def f(U,V,inst):
    """
    
    Parameters
    ----------
    U : r x m array
    V : r x n array
    inst : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    A = inst['A']
    
    return 1/2*norm(A - U.T@V)**2



def F(U,V,inst):
    """
    
    Parameters
    ----------
    U : r x m array
    V : r x n array
    inst : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    UV = U.T@V
    
    if sum(sum(U<0))>0 or sum(sum(U>1))>0 or sum(sum(V<0))>0 or sum(sum(U>1))>0:
        
        return np.inf, UV
        
    else:
    
        A = inst['A']
        
        
            
        return 1/2*norm(A - UV)**2, UV
    
def FU(ui,V,UV,ind,lamd,inst):
    """
    

    Parameters
    ----------
    U : r x m array
    V : r x n array
    ind : index 
    d : displacement vector
    inst : TYPE

    Returns
    -------
    None.

    """
    
    if sum((ui + lamd[:,ind])<0)>0 or  sum((ui + lamd[:,ind])>1)>0:
        return np.inf
    
    else:
        
        A = inst['A']
        
        return 1/2*norm(A-UV-lamd.T@V)**2
    
def FUc(ui,V,UV,ind,lam,d,inst):
    """
    

    Parameters
    ----------
    U : r x m array
    V : r x n array
    ind : index 
    d : displacement vector
    inst : TYPE

    Returns
    -------
    None.

    """
    lamd = (1+lam)*d
    
    if sum((ui + lamd[:,ind])<0)>0 or  sum((ui + lamd[:,ind])>1)>0:
        return np.inf
    
    else:
        
        A = inst['A']
        
        return 1/2*norm(A-UV-lamd.T@V)**2
    
def FV(U,vi,UV,ind,lamd,inst):
    """
    

    Parameters
    ----------
    U : r x m array
    V : r x n array
    ind : index 
    d : displacement vector
    inst : TYPE

    Returns
    -------
    None.

    """
        
    if sum((vi + lamd[:,ind])<0) > 0 or sum((vi + lamd[:,ind])>1) > 0:
        
        return np.inf
    
    else:
        
        A = inst['A']
        
        return 1/2*norm(A-UV-U.T@lamd)**2
    
    
def FVc(U,vi,UV,ind,lam,d,inst):
    """
    

    Parameters
    ----------
    U : r x m array
    V : r x n array
    ind : index 
    d : displacement vector
    inst : TYPE

    Returns
    -------
    None.

    """
    
    lamd = (1+lam)*d
    
    if sum((vi + lamd[:,ind])<0) > 0 or sum((vi + lamd[:,ind])>1) > 0:
        
        return np.inf
    
    else:
        
        A = inst['A']
        
        return 1/2*norm(A-UV-U.T@lamd)**2
    





" Main algorithms:"


"Nonboosted variants. Include the possibility of conducting a nonmonotone linesearch."
"We distinguish various posibilities for setting the initial trial stepsize:"

"With bar tau fixed at the beginning of each iteration"
def RNBPG_btau_fixed(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    
    tauU = btau
    
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            tauU =  btau
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                if norm(di) != 0:
                    
                                        
                    temp_M = min(M,it+1)
                    if FU(ui,V,UV,ik,d,inst) <= max(F_list[-temp_M:]) - sigma*norm(d)**2:

                        Unew = U + d
                                                
                        adaptive_ls = False # the adaptive_ls succeedded
                    else: #we decrease the stepsize (backtracking)
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                        
               
            # Update:
            F_new, UV = F(Unew,V,inst)
            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new
            
                 
        # Update V:
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV = btau #max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                   
                    
                    temp_M = min(M,it+1)
                    if FV(U,vi,UV,jk,d,inst) <= max(F_list[-temp_M:]) - sigma*norm(d)**2:

                        Vnew = V + d
                        
                        adaptive_ls = False
                    else:
                        tauV = beta*tauV
                
                else: # the direction is zero
                    Vnew = V.copy()
                    break
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        
        
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False

    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution


"With bar tau decreasing along the iterations at the beginning of each iteration"
def RNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    
    tauU = btau
    
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            tauU =  max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                if norm(di) != 0:
                    
                                        
                    temp_M = min(M,it+1)
                    if FU(ui,V,UV,ik,d,inst) <= max(F_list[-temp_M:]) - sigma*norm(d)**2:

                        Unew = U + d
                                                
                        adaptive_ls = False # the adaptive_ls succeedded
                    else: #we decrease the stepsize (backtracking)
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                        
               
            # Update:
            F_new, UV = F(Unew,V,inst)
            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new
            
                 
        # Update V:
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV = max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                   
                    
                    temp_M = min(M,it+1)
                    if FV(U,vi,UV,jk,d,inst) <= max(F_list[-temp_M:]) - sigma*norm(d)**2:

                        Vnew = V + d
                        
                        adaptive_ls = False
                    else:
                        tauV = beta*tauV
                
                else: # the direction is zero
                    Vnew = V.copy()
                    break
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        
        
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False

    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution

"With bar tau adaptive each iteration"
def RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    
    
    tauU = btau
        
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            tauU =  max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                if norm(di) != 0:
                    
                                        
                    temp_M = min(M,it+1)
                    if FU(ui,V,UV,ik,d,inst) <= max(F_list[-temp_M:]) - sigma*norm(d)**2:
                        
                        Unew = U + d
                        
                        if counter_adaptive == 0:
                            tauU = 1/beta*tauU
                                                
                        adaptive_ls = False # the adaptive_ls succeedded
                    else: #we decrease the stepsize (backtracking)
                        counter_adaptive += 1
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                        
               
            # Update:
            F_new, UV = F(Unew,V,inst)
            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new
            
                 
        # Update V:
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV = max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                   
                    
                    temp_M = min(M,it+1)
                    if FV(U,vi,UV,jk,d,inst) <= max(F_list[-temp_M:]) - sigma*norm(d)**2:

                        Vnew = V + d
                        
                        if counter_adaptive == 0:
                            tauV = 1/beta*tauV
                        
                        adaptive_ls = False
                    else:
                        counter_adaptive += 1
                        tauV = beta*tauV
                        
                
                else: # the direction is zero
                    Vnew = V.copy()
                    break
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        
        
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        if stop_rule < 2:
        
            if error_crit <= prec or counter[stop_rule] >= max_stop:
                continua = False
        
        else:
            
            if counter[0]>=1500000 or F_new <= max_stop:
                continua = False
                
            

    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution
            
            
"Boosted versions:"

"With bar tau fixed at the beginning of each iteration"
def BRNBPG_btau_fixed(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)

    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =  btau#max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                        #Uhat = U + d
                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        b_counter = 0 
                        
                        while lam > 1 and FU(ui,V,UV,ik,lam*d,inst) > FUhat - alpha*(lam-1)**2*norm(di)**2:
                            lam = rho*lam
                            b_counter += 1
                            
                        if lam > 1:
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                        else:
                            blam = blam0
                            lam = 1
                        
                        Unew = U + lam*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV = btau # max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:
                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        b_counter = 0
                        
                        while lam > 1 and FV(U,vi,UV,jk,lam*d,inst)  > FVhat - alpha*(lam-1)**2*norm(di)**2:
                            lam  = rho*lam
                            b_counter += 1
                            
                        if lam > 1:
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                        else:
                            blam = blam0
                            lam = 1
                            
                        Vnew = V + lam*d
                        
                    else: #adapt the stepsize
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]

    
    return solution


"With bar tau fixed at the beginning of each iteration"
"Boosted linesearch is not adaptive"
def BRNBPG_btau_fixed_nonadapt(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =  btau#max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        
                        while lam > 1 and FU(ui,V,UV,ik,lam*d,inst) > FUhat - alpha*(lam-1)**2*norm(di)**2:
                            lam = rho*lam
                            
                                
                        else:
                            blam = blam0
                            lam = 1
                        
                        Unew = U + lam*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV = btau # max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:

                        
                        #Vhat = V + d  
                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        
                        while lam > 1 and FV(U,vi,UV,jk,lam*d,inst)  > FVhat - alpha*(lam-1)**2*norm(di)**2:
                            lam  = rho*lam
                            

                                
                        else:
                            blam = blam0
                            lam = 1
                            
                        Vnew = V + lam*d
                        
                    else: #adapt the stepsize
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution


"With bar tau decreasing along the iterations at the beginning of each iteration"
def BRNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =   max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                        
                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        b_counter = 0 
                        
                        while lam > 1 and FU(ui,V,UV,ik,lam*d,inst) > FUhat - alpha*(lam-1)**2*norm(di)**2:
                            lam = rho*lam
                            b_counter += 1
                            
                        if lam > 1:
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                        else:
                            blam = blam0
                            lam = 1
                        
                        Unew = U + lam*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV =  max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:                        
                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        b_counter = 0
                        
                        while lam > 1 and FV(U,vi,UV,jk,lam*d,inst)  > FVhat - alpha*(lam-1)**2*norm(di)**2:
                            lam  = rho*lam
                            b_counter += 1
                            
                        if lam > 1:
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                        else:
                            blam = blam0
                            lam = 1
                            
                        Vnew = V + lam*d
                        
                    else: #adapt the stepsize
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution


"With bar tau decreasing along the iterations at the beginning of each iteration"
"The boosted linesearch is not self-adaptive"
def BRNBPG_btau_dec_nonadapt(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =   max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        
                        while lam > 1 and FU(ui,V,UV,ik,lam*d,inst) > FUhat - alpha*(lam-1)**2*norm(di)**2:
                            lam = rho*lam
                            
                                
                        else:
                            blam = blam0
                            lam = 1
                        
                        Unew = U + lam*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV =  max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:
                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        
                        while lam > 1 and FV(U,vi,UV,jk,lam*d,inst)  > FVhat - alpha*(lam-1)**2*norm(di)**2:
                            lam  = rho*lam
                            

                                
                        else:
                            blam = blam0
                            lam = 1
                            
                        Vnew = V + lam*d
                        
                    else: #adapt the stepsize
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution


"With bar tau adaptive each iteration but lambda is not adaptive"
def BRNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =   max(min(tau_max,tauU ),tau_min)#max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                        #Uhat = U + d
                        if  counter_adaptive == 0:
                            tauU = 1/beta*tauU

                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        
                        while lam > 1 and FU(ui,V,UV,ik,lam*d,inst) > FUhat - alpha*(lam-1)**2*norm(di)**2:
                            lam = rho*lam
                            
                        else:
                            blam = blam0
                            lam = 1
                        
                        Unew = U + lam*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        counter_adaptive += 1
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV =  max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:
                        if counter_adaptive ==  0:
                            tauV = 1/beta*tauV
                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        
                        while lam > 1 and FV(U,vi,UV,jk,lam*d,inst)  > FVhat - alpha*(lam-1)**2*norm(di)**2:
                            lam  = rho*lam
                            
                                
                        else:
                            blam = blam0
                            lam = 1
                            
                        Vnew = V + lam*d
                        
                    else: #adapt the stepsize
                        counter_adaptive += 1
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution

"With bar tau and lambda adaptive at each iteration"
def BRNBPG_btau_double_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =   max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                        if  counter_adaptive == 0:
                            tauU = 1/beta*tauU

                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        b_counter = 0 
                        
                        while lam > 1 and FU(ui,V,UV,ik,lam*d,inst) > FUhat - alpha*(lam-1)**2*norm(di)**2:
                            lam = rho*lam
                            b_counter += 1
                            
                        if lam > 1:
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                        else:
                            blam = blam0
                            lam = 1
                        
                        Unew = U + lam*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        counter_adaptive += 1
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV =  max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:
                        
                        if counter_adaptive ==  0:
                            tauV = 1/beta*tauV

                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        b_counter = 0
                        
                        while lam > 1 and FV(U,vi,UV,jk,lam*d,inst)  > FVhat - alpha*(lam-1)**2*norm(di)**2:
                            lam  = rho*lam
                            b_counter += 1
                            
                        if lam > 1:
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                        else:
                            blam = blam0
                            lam = 1
                            
                        Vnew = V + lam*d
                        
                    else: #adapt the stepsize
                        counter_adaptive += 1
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)
            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA
        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution


"With bar tau and lambda adaptive at each iteration"
def BRNBPG_btau_double_adaptive_constrained(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap=50,prec=1e-6,stop_rule=0,max_stop=1000):
    
    A = inst['A']
    m = inst['m']
    n = inst['n']
    
    normA = norm(A)
    
    
    U = U0.copy()
    V = V0.copy()
    
    F_old, UV_old = F(U,V,inst)
    
    UV = UV_old.copy()
    
    solution = {}
    
    F_list = [F_old]
    
    counter = [0,0]
    
    t0 = time()
    
    blam = blam0
    
    
    tauU = btau
    
    tauV = btau
    
    continua = True
    it = 0
    while continua:
        
        ik = random.randint(0,n+m)
        
        " Update U:"
        if ik < m:
            ui = U[:,ik]
            
            
            tauU =   max(min(tau_max,tauU ),tau_min)
            
            gf = -V@(A.T[:,ik] - V.T@ui)
            d = np.zeros(U.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_ui = ui - tauU*gf
                
                di =  - ui + np.maximum(temp_ui,0.)
                
                
                d[:,ik] = di
                
                
                if norm(di) != 0:
                    
                    FUhat = FU(ui,V,UV,ik,d,inst)
                    if FUhat <= F_list[-1] - sigma*norm(d)**2:
                
                        if  counter_adaptive == 0:
                            tauU = 1/beta*tauU

                        adaptive_ls = False # the adaptive_ls succeedded
                    
                        
                        # Once the nonmonotone linesearch finished we do the boosted one:
                        lam = blam
                        b_counter = 0 
                        
                        index_hat = ((ui+di)==0) | ((ui+di)==1)
                        
                        if sum(di[index_hat]) == 0:
                            
                        
                            while b_counter < 2 and FUc(ui,V,UV,ik,lam,d,inst) > FUhat - alpha*(lam)**2*norm(di)**2:
                                lam = rho*lam
                                b_counter += 1
                                
              
                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                            if b_counter == 2:
                                lam = 0
                                    
                                
                        else: 
                            lam = 0
                        
                        Unew = U + (1+lam)*d
                                                                                
                    else: #we decrease the stepsize (backtracking)
                        counter_adaptive += 1
                        tauU = beta*tauU
                        
                else: #the direction is zero
                    Unew = U.copy()
                    break  # end the while loop
                
            # Update:
            F_new, UV = F(Unew,V,inst)
            

            F_list = np.append(F_list,F_new)
            
            U = Unew.copy()
            F_old = F_new    
                
         
        # Update V: 
        elif ik >= m:
            jk = ik-m
            vi = V[:,jk]
            
            
            tauV =  max(min(tau_max,tauV ),tau_min)

            gf = -U@(A[:,jk] - U.T@vi)
            
            d = np.zeros(V.shape)
            
            adaptive_ls = True
            counter_adaptive = 0
            while adaptive_ls:
                
                temp_vi = vi - tauV*gf
                
                di = -vi + np.maximum(temp_vi,0.)
                d[:,jk] = di
                
                if norm(di) != 0:
                    
                    FVhat = FV(U,vi,UV,jk,d,inst)
                    if FVhat <= F_list[-1] - sigma*norm(d)**2:
                        
                        if counter_adaptive ==  0:
                            tauV = 1/beta*tauV

                        adaptive_ls = False
                        
                        # Once the nonmonotone linesearch finished we do the boosted one
                        lam = blam
                        b_counter = 0
                        
                        index_hat = ((vi+di)==0) | ((vi+di)==1)
                                                
                        if sum(di[index_hat]) == 0:
                            
                        
                            while b_counter < 2 and  FVc(U,vi,UV,jk,lam,d,inst)  > FVhat - alpha*(lam)**2*norm(di)**2:
                                lam  = rho*lam
                                b_counter += 1
                                

                            if b_counter == 0:
                                blam = 2*blam
                            else:
                                blam = max(blam0,lam)
                                
                            
                            if b_counter == 2:
                                lam = 0
                                
                                    
                        else:
                            lam = 0
                            
                        Vnew = V + (1+lam)*d
                        
                    else: #adapt the stepsize
                        counter_adaptive += 1
                        tauV = beta*tauV
                        
                else: # the direction is zero
                    Vnew = V.copy()
                    
                    
                
            # Update:
            F_new, UV = F(U,Vnew,inst)

            F_list = np.append(F_list,F_new)
            
            V = Vnew.copy()
            F_old = F_new
            

         
         
        # Check stopping condition:
            
        it += 1
        counter[0] += 1
        counter[1] = time() - t0
        
        error_crit = abs(F_list[-min(gap,it+1)] - F_new)/normA

        
        
        if error_crit <= prec or counter[stop_rule] >= max_stop:
            continua = False
            

    
    solution['U'] = U
    solution['V'] = V
    solution['F'] = F_new
    solution['F_list'] = F_list
    solution['it'] = counter[0]
    solution['time'] = counter[1]
    
    return solution
