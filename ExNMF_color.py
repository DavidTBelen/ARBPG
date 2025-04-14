# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:29:11 2025

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper: Enhanced randomized block proximal method with locally
Lipschitz continuous gradient
"""

import numpy as np
from numpy import random
from Algorithms import NMF_adaptive as NMF
from PIL import Image
import matplotlib.pyplot as plt
from Auxiliaries import DisplayResultsNMF as DRNMF





#theSeed = np.random.choice(20000,1)[0]
theSeed = 1

print('Seed: ',theSeed)
random.seed(theSeed)

images = [ 'Atacama_color', 'Santiago_color', 'Valdivia_color', 'Niebla_color']

rs = [100, 100, 100, 150]

output_iterations = np.zeros([len(images),3,3]) # images, rgb, alg
output_time = np.zeros([len(images),3,3])
output_Fval = np.zeros([len(images),3,3])
output_PSNR = np.zeros([len(images),3,3]) #peak signal-to-noise ratio
    
it_matching_boosted = np.zeros([len(images),3])
time_matching_boosted = np.zeros([len(images),3])
fval_matching_boosted = np.zeros([len(images),3])
PSNR_matching_boosted = np.zeros([len(images),3])


for ii in range(len(images)):
    
    r = rs[ii] # rank of the compression
    
    sel_image = images[ii]
    fig_name = 'Figures/'+sel_image+'.jpg'
    Im = plt.imread(fig_name)
    IR = Im[:,:,0]
    IG = Im[:,:,1]
    IB = Im[:,:,2]
    
    IR = IR.astype(float)/255
    IB = IB.astype(float)/255
    IG = IG.astype(float)/255
    
    
    # IR_im = (255.0 * IR).astype(np.uint8)
    # IG_im = (255.0 * IG).astype(np.uint8)
    # IB_im = (255.0 * IB).astype(np.uint8)
    # I_im = np.concatenate((IR_im[...,None],IG_im[...,None],IB_im[...,None]),axis=2)
    # I_im = Image.fromarray(I_im)
    # I_im.save('Figures/color/test.jpg')
    
    
        
    
    
    
    # For saving the image:
    # IR_im_0 = (255.0 * IR).astype(np.uint8)
    # IR_im = Image.fromarray(IR_im_0)
    # IR_im.save('Figures/'+sel_image+'_red.jpg')
    
    
    
    m, n = IR.shape
    
    
    "Algorithmic parameters"
    
    # Boosted RNBPG parameters:
    M = 10
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
    max_stop = 1500000
    
    btau = 2
    

    
    
    " Red images"
    
    
    print('\n ### Image:', sel_image, 'Color:', 'Red')
    
    inst = {}

    inst['A'] = IR
    inst['m'] = m
    inst['n'] = n
    inst['r'] = r
    

    
    U0 = np.random.random([r,m])
    V0 = np.random.random([r,n])

    
    "Nonmonotone:" 
    sol = NMF.RNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Randomized Nonmonotone BCD:')
    print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
    print('Fval=',sol['F'])
    
    UN = sol['U']
    VN = sol['V']
    
    IRN = UN.T@VN
    
    
    IRN_im_0 = (255.0 * IRN).astype(np.uint8)
    IRN_im = Image.fromarray(IRN_im_0)

    psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRN-IR,'fro')**2))
    print('PSNR=',psnr)
    
    output_iterations[ii,0,0] = sol['it']
    output_time[ii,0,0] = np.round(sol['time'],2)
    output_Fval[ii,0,0] = np.round(sol['F'],2)
    output_PSNR[ii,0,0] =psnr  #peak signal-to-noise ratio
            

    
    "Monotone"
    solB = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Randomized Monotone BCD:')
    print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
    print('Fval=',solB['F'])
    
    UB = solB['U']
    VB = solB['V']
    
    
    
    IRB = UB.T@VB
    
    IRB_im_0 = (255.0 * IRB).astype(np.uint8)
    IRB_im = Image.fromarray(IRB_im_0)
    
    psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRB-IR,'fro')**2))
    print('PSNR=',psnr)

    
    output_iterations[ii,0,1] = solB['it']
    output_time[ii,0,1] = np.round(solB['time'],2)
    output_Fval[ii,0,1] = np.round(solB['F'],2)
    output_PSNR[ii,0,1] =psnr  #peak signal-to-noise ratio
    
    
    "Boosted Monotone"
    solM = NMF.BRNBPG_btau_double_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Boosted Randomized Monotone BCD:')
    print(np.round(solM['time'],2), 'seconds ',solM['it'], 'iterations')
    print('Fval=',solM['F'])
    
    UM = solM['U']
    VM = solM['V']
    

    
    IRM = UM.T@VM
    
    IRM_im_0 = (255.0 * IRM).astype(np.uint8)
    IRM_im = Image.fromarray(IRM_im_0)
    
    # psnr = 10*np.log10( 255**2*m*n/np.linalg.norm(IR_im_0-IRM_im_0)**2)
    psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRM-IR,'fro')**2))
    print('PSNR=',psnr)

    
    output_iterations[ii,0,2] = solM['it']
    output_time[ii,0,2] = np.round(solM['time'],2)
    output_Fval[ii,0,2] = np.round(solM['F'],2)
    output_PSNR[ii,0,2] =psnr  #peak signal-to-noise ratio
    
    "Checking how long it take the monotone to achieve the same Fval than the boosted:"
    
    solF = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=2,max_stop=solM['F'])
    print('\n Monotone until boosted is reached:')
    print(np.round(solF['time'],2), 'seconds ',solF['it'], 'iterations')
    print('Fval=',solF['F'])
    
    UF = solF['U']
    VF = solF['V']
    
    
    
    IRF = UF.T@VF
    
    IRF_im_0 = (255.0 * IRF).astype(np.uint8)
    IRF_im = Image.fromarray(IRF_im_0)
    
    psnr = 10*np.log10(np.max(IR)**2*m*n/(np.linalg.norm(IRF-IR,'fro')**2))
    print('PSNR=',psnr)

    
    it_matching_boosted[ii,0] = solF['it']
    time_matching_boosted[ii,0] = np.round(solF['time'],2)
    fval_matching_boosted[ii,0] = np.round(solF['F'],2)
    PSNR_matching_boosted[ii,0] = psnr  #peak signal-to-noise ratio
        
    
    
    
    " Green images"
    print('\n ### Image:', sel_image, 'Color:', 'Green')

    inst = {}
    
    inst['A'] = IG
    inst['m'] = m
    inst['n'] = n
    inst['r'] = r


    U0 = np.random.random([r,m])
    V0 = np.random.random([r,n])

    "Non-boosted:" 
    sol = NMF.RNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Randomized Nonmonotone BCD:')
    print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
    print('Fval=',sol['F'])
    
    UN = sol['U']
    VN = sol['V']
    
    IGN = UN.T@VN
    
    
    IGN_im_0 = (255.0 * IGN).astype(np.uint8)
    IGN_im = Image.fromarray(IGN_im_0)
    
    psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGN-IG,'fro')**2))
    print('PSNR=',psnr)
    
    output_iterations[ii,1,0] = sol['it']
    output_time[ii,1,0] = np.round(sol['time'],2)
    output_Fval[ii,1,0] = np.round(sol['F'],2)
    output_PSNR[ii,1,0] =psnr  #peak signal-to-noise ratio
            

    
    "Monotone"
    solB = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n  Randomized Monotone BCD:')
    print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
    print('Fval=',solB['F'])
    
    UB = solB['U']
    VB = solB['V']
    
    
    
    IGB = UB.T@VB
    
    IGB_im_0 = (255.0 * IGB).astype(np.uint8)
    IGB_im = Image.fromarray(IGB_im_0)

    psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGB-IG,'fro')**2))

    print('PSNR=',psnr)

    
    output_iterations[ii,1,1] = solB['it']
    output_time[ii,1,1] = np.round(solB['time'],2)
    output_Fval[ii,1,1] = np.round(solB['F'],2)
    output_PSNR[ii,1,1] =psnr  #peak signal-to-noise ratio
    
    
    "Boosted Monotone"
    solM = NMF.BRNBPG_btau_double_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Boosted Randomized Monotone BCD:')
    print(np.round(solM['time'],2), 'seconds ',solM['it'], 'iterations')
    print('Fval=',solM['F'])
    
    UM = solM['U']
    VM = solM['V']
    

    
    IGM = UM.T@VM
    
    IGM_im_0 = (255.0 * IGM).astype(np.uint8)
    IGM_im = Image.fromarray(IGM_im_0)
    
    psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGM-IG,'fro')**2))
    print('PSNR=',psnr)

    
    output_iterations[ii,1,2] = solM['it']
    output_time[ii,1,2] = np.round(solM['time'],2)
    output_Fval[ii,1,2] = np.round(solM['F'],2)
    output_PSNR[ii,1,2] =psnr  #peak signal-to-noise ratio
    
    
    "Checking how long it take the monotone to achieve the same Fval than the boosted:"
    
    solF = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=2,max_stop=solM['F'])
    print('\n Monotone until boosted is reached:')
    print(np.round(solF['time'],2), 'seconds ',solF['it'], 'iterations')
    print('Fval=',solF['F'])
    
    UF = solF['U']
    VF = solF['V']
    
    
    
    IGF = UF.T@VF
    
    IGF_im_0 = (255.0 * IGF).astype(np.uint8)
    IGF_im = Image.fromarray(IGF_im_0)
    
    psnr = 10*np.log10(np.max(IG)**2*m*n/(np.linalg.norm(IGF-IG,'fro')**2))
    print('PSNR=',psnr)

    
    it_matching_boosted[ii,1] = solF['it']
    time_matching_boosted[ii,1] = np.round(solF['time'],2)
    fval_matching_boosted[ii,1] = np.round(solF['F'],2)
    PSNR_matching_boosted[ii,1] = psnr  #peak signal-to-noise ratio
    
    
    " Blue images"
    print('\n ### Image:', sel_image, 'Color:', 'Blue')

    inst = {}
    
    inst['A'] = IB
    inst['m'] = m
    inst['n'] = n
    inst['r'] = r


    U0 = np.random.random([r,m])
    V0 = np.random.random([r,n])

    "Non-boosted:" 
    sol = NMF.RNBPG_btau_dec(U0,V0,inst,btau,tau_min,tau_max,M,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Randomized Nonmonotone BCD:')
    print(np.round(sol['time'],2), 'seconds ',sol['it'], 'iterations')
    print('Fval=',sol['F'])
    
    UN = sol['U']
    VN = sol['V']
    
    IBN = UN.T@VN
    
    
    IBN_im_0 = (255.0 * IBN).astype(np.uint8)
    IBN_im = Image.fromarray(IBN_im_0)
    
    psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBN-IB,'fro')**2))
    print('PSNR=',psnr)
    
    output_iterations[ii,2,0] = sol['it']
    output_time[ii,2,0] = np.round(sol['time'],2)
    output_Fval[ii,2,0] = np.round(sol['F'],2)
    output_PSNR[ii,2,0] =psnr  #peak signal-to-noise ratio
            

    
    "Monotone"
    solB = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n  Randomized Monotone BCD:')
    print(np.round(solB['time'],2), 'seconds ',solB['it'], 'iterations')
    print('Fval=',solB['F'])
    
    UB = solB['U']
    VB = solB['V']
    
    
    
    IBB = UB.T@VB
    
    IBB_im_0 = (255.0 * IBB).astype(np.uint8)
    IBB_im = Image.fromarray(IBB_im_0)

    psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBB -IB,'fro')**2))

    print('PSNR=',psnr)

    
    output_iterations[ii,2,1] = solB['it']
    output_time[ii,2,1] = np.round(solB['time'],2)
    output_Fval[ii,2,1] = np.round(solB['F'],2)
    output_PSNR[ii,2,1] =psnr  #peak signal-to-noise ratio
    
    
    "Boosted Monotone"
    solM = NMF.BRNBPG_btau_double_adaptive(U0,V0,inst,btau,tau_min,tau_max,blam0,alpha,rho,sigma,beta,gap,prec,stop_rule=0,max_stop=max_stop)
    print('\n Boosted Randomized Monotone BCD:')
    print(np.round(solM['time'],2), 'seconds ',solM['it'], 'iterations')
    print('Fval=',solM['F'])
    
    UM = solM['U']
    VM = solM['V']
    

    
    IBM = UM.T@VM
    
    IBM_im_0 = (255.0 * IBM).astype(np.uint8)
    IBM_im = Image.fromarray(IBM_im_0)
    
    # psnr = 10*np.log10( 255**2*m*n/np.linalg.norm(IR_im_0-IRM_im_0)**2)
    psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBM-IB,'fro')**2))
    print('PSNR=',psnr)

    
    output_iterations[ii,2,2] = solM['it']
    output_time[ii,2,2] = np.round(solM['time'],2)
    output_Fval[ii,2,2] = np.round(solM['F'],2)
    output_PSNR[ii,2,2] =psnr  #peak signal-to-noise ratio
    
    
    "Checking how long it take the monotone to achieve the same Fval than the boosted:"
    
    solF = NMF.RNBPG_btau_adaptive(U0,V0,inst,btau,tau_min,tau_max,1,sigma,beta,gap,prec,stop_rule=2,max_stop=solM['F'])
    print('\n Monotone until boosted is reached:')
    print(np.round(solF['time'],2), 'seconds ',solF['it'], 'iterations')
    print('Fval=',solF['F'])
    
    UF = solF['U']
    VF = solF['V']
    
    
    
    IBF = UF.T@VF
    
    IBF_im_0 = (255.0 * IBF).astype(np.uint8)
    IBF_im = Image.fromarray(IBF_im_0)
    
    psnr = 10*np.log10(np.max(IB)**2*m*n/(np.linalg.norm(IBF-IB,'fro')**2))
    print('PSNR=',psnr)

    
    it_matching_boosted[ii,2] = solF['it']
    time_matching_boosted[ii,2] = np.round(solF['time'],2)
    fval_matching_boosted[ii,2] = np.round(solF['F'],2)
    PSNR_matching_boosted[ii,2] = psnr  #peak signal-to-noise ratio
    
    
    
    
    " Saving the images: "
    
    IN_im = np.concatenate((IRN_im_0[...,None],IGN_im_0[...,None],IBN_im_0[...,None]),axis=2)
    IN_im = Image.fromarray(IN_im)
    IN_im.save('Figures/color/'+sel_image+'_N.jpg')
    
    IBo_im = np.concatenate((IRB_im_0[...,None],IGB_im_0[...,None],IBB_im_0[...,None]),axis=2)
    IBo_im = Image.fromarray(IBo_im)
    IBo_im.save('Figures/color/'+sel_image+'_B.jpg')
    
    IM_im = np.concatenate((IRM_im_0[...,None],IGM_im_0[...,None],IBM_im_0[...,None]),axis=2)
    IM_im = Image.fromarray(IM_im)
    IM_im.save('Figures/color/'+sel_image+'_M.jpg')
    
    np.savez('Results/Results_ExNMF_color', output_iterations, output_time, 
             output_Fval, output_PSNR, it_matching_boosted, time_matching_boosted,
             fval_matching_boosted, PSNR_matching_boosted) 
    



"Compute average value over colors for the  table"

output_iterations_mean = np.mean(output_iterations, axis=1)
output_time_mean = np.round(np.mean(output_time,axis=1),2)
output_Fval_mean = np.round(np.mean(output_Fval,axis=1),2)
output_PSNR_mean = np.round(np.mean(output_PSNR,axis=1),2)



fname = "Results/Table_ExNMF_color"
DRNMF.generate_table(fname, images, output_iterations_mean, output_time_mean, output_Fval_mean, output_PSNR_mean)


# "Load data" 
# npzfile = np.load('Results/Results_ExNMF_color.npz',allow_pickle = True )
# # output_iterations = npzfile['arr_0']
# # output_time = npzfile['arr_1']
# # output_Fval = npzfile['arr_2']
# # output_PSNR = npzfile['arr_3']
# output_iterations = npzfile['arr_0']
# output_time = npzfile['arr_1']
# it_matching_boosted = npzfile['arr_4']
# time_matching_boosted = npzfile['arr_5']

" Plot comparison between boosted and nonboosted:"

wr_it = it_matching_boosted[:,0] / output_iterations[:,0,2]
wg_it = it_matching_boosted[:,1] / output_iterations[:,1,2]
wb_it = it_matching_boosted[:,2] / output_iterations[:,2,2]

wr_time = time_matching_boosted[:,0] / output_time[:,0,2]
wg_time = time_matching_boosted[:,1] / output_time[:,1,2]
wb_time = time_matching_boosted[:,2] / output_time[:,2,2]

xticks = np.array([1,2,3,4])


plt.figure(figsize=(8, 5))
#plt.xlabel('$k$')
plt.ylabel('Iterations ARBPG/ Iterations ARBPG-B')
plt.scatter(xticks,wr_it,facecolors='None',color='r',s=100,linewidth=2,label='red')
plt.scatter(xticks,wg_it,facecolors='None',marker='d',s=100,linewidth=2,color='g',label='green')
plt.scatter(xticks,wb_it,facecolors='None',marker='X',s=100,linewidth=2,color = 'b',label='blue')
plt.xticks([1,2,3,4],['Atacama', 'Santiago', 'Valdivia', 'Niebla'])

#plt.plot(v1,u[0,:], '.',color='r')
plt.legend(loc='best')
plt.savefig('Figures/color/iterations_comp_RBPG.pdf',bbox_inches='tight',dpi=800)


plt.figure(figsize=(8, 5))
#plt.xlabel('$k$')
plt.ylabel('Time ARBPG/ Time ARBPG-B')
plt.scatter(xticks,wr_time,facecolors='None',color='r',s=100,linewidth=2,label='red')
plt.scatter(xticks,wg_time,facecolors='None',marker='d',s=100,linewidth=2,color='g',label='green')
plt.scatter(xticks,wb_time,facecolors='None',marker='X',s=100,linewidth=2,color = 'b',label='blue')
plt.xticks([1,2,3,4],['Atacama', 'Santiago', 'Valdivia', 'Niebla'])

#plt.plot(v1,u[0,:], '.',color='r')
plt.legend(loc='best')
plt.savefig('Figures/color/time_comp_RBPG.pdf',bbox_inches='tight',dpi=800)

