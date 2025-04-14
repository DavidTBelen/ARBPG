# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:05:29 2025

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper: Randomized block proximal method with locally
Lipschitz continuous gradient
"""

def generate_table(fname, vranks, out_it, out_time, out_Fval, out_PSNR):
    """

    Parameters
    ----------
    fname : name of the output file
    algorithms: labels for the algorithms
    summary : np.array([len(data, algorithm, headers])
            algorithms = QN x DCA x BDCA x RCNM (in that order)

    Returns
    -------
    fname.txt: file with latex code for table

    """
    
    with open(fname, 'w') as f:
        #f.write(tabulate(summary, headers='keys', tablefmt='latex'))
        
        table_format = r"\begin{tblr}{width = \linewidth,colspec = {QQS[table-format=6,detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=2.2, detect-weight,  mode=text]},}"
        
        f.write(table_format)
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        
        f.write(r"{{{Rank}}} & Method & {{{\# Iterations}}} & {{{Time (s.)}}} & {{{$F(U^{out},V^{out})$}}} & {{{PSNR}}} \\")
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        for dd in range(len(vranks)):
            
            
            f.write(r'& RNBPG &'+str(int(out_it[dd,0]))+' & '+str(out_time[dd,0])+
                            ' & '+str(round(out_Fval[dd,0],2))+ '&' +str(round(out_PSNR[dd,0],2))+  r'\\')
            
            f.write('\n')
            
            f.write(str(vranks[dd]))

            
            f.write(r'& ARBPG &'+str(int(out_it[dd,1]))+' & '+str(out_time[dd,1])+
                ' & '+str(round(out_Fval[dd,1],2))+ '&' +str(round(out_PSNR[dd,1],2))+  r'\\')


            
            f.write('\n')
            
            f.write(r'& ARBPG-B &'+str(int(out_it[dd,2]))+' & '+str(out_time[dd,2])+
                            ' & '+str(round(out_Fval[dd,2],2))+ '&' +str(round(out_PSNR[dd,2],2))+  r'\\')
            
            f.write('\n')
            
            f.write("\\hline")
            
        f.write('\n')
        f.write("\\end{tblr}")
    
def table_strategies_MN(fname, out_it, out_time, out_Fval):
    """

    Parameters
    ----------
    fname : name of the output file
    algorithms: labels for the algorithms
    summary : np.array([len(data, algorithm, headers])
            algorithms = QN x DCA x BDCA x RCNM (in that order)

    Returns
    -------
    fname.txt: file with latex code for table

    """
    
    with open(fname, 'w') as f:
        #f.write(tabulate(summary, headers='keys', tablefmt='latex'))
        
        table_format = r"\begin{tblr}{width = \linewidth,colspec = {QS[table-format=6,detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]},}"
        
        f.write(table_format)
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        
        f.write(r" {{{Strategy for $\bar{\tau}_k$}}} & {{{\# Iterations}}} & {{{Time (s.)}}} & {{{$F(U^{out},V^{out})$}}} \\")
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
            
            
        f.write(r'Fixed  &'+str(int(out_it[0]))+' & '+str(out_time[0])+
                        ' & '+str(round(out_Fval[0],2))+   r'\\')
        
        f.write('\n')
        
        
        f.write(r' Decreasing  &'+str(int(out_it[1]))+' & '+str(out_time[1])+
                        ' & '+str(round(out_Fval[1],2))+   r'\\')
        
        f.write('\n')
        
        f.write(r' Adaptive  &'+str(int(out_it[2]))+' & '+str(out_time[2])+
                        ' & '+str(round(out_Fval[2],2))+  r'\\')
        
        f.write('\n')
        
        f.write("\\hline")
            
        f.write('\n')
        f.write("\\end{tblr}")
        

def table_strategies_B(fname, out_it, out_time, out_Fval):
    """

    Parameters
    ----------
    fname : name of the output file
    algorithms: labels for the algorithms
    summary : np.array([len(data, algorithm, headers])
            algorithms = QN x DCA x BDCA x RCNM (in that order)

    Returns
    -------
    fname.txt: file with latex code for table

    """
    
    with open(fname, 'w') as f:
        #f.write(tabulate(summary, headers='keys', tablefmt='latex'))
        
        table_format = r"\begin{tblr}{width = \linewidth,colspec = {QQS[table-format=6,detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]},}"
        
        f.write(table_format)
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        
        f.write(r" {{{Strategy for $\bar{\lambda}_k$}}} & {{{Strategy for $\bar{\tau}_k$}}} & {{{\# Iterations}}} & {{{Time (s.)}}} & {{{$F(U^{out},V^{out})$}}} \\")
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
            
            
        f.write(r'& Fixed  &'+str(int(out_it[0]))+' & '+str(out_time[0])+
                        ' & '+str(round(out_Fval[0],2))+   r'\\')
        
        f.write('\n')
        
        
        f.write(r'Fixed & Decreasing  &'+str(int(out_it[2]))+' & '+str(out_time[2])+
                        ' & '+str(round(out_Fval[2],2))+   r'\\')
        
        f.write('\n')
        
        f.write(r'& Adaptive  &'+str(int(out_it[4]))+' & '+str(out_time[4])+
                        ' & '+str(round(out_Fval[4],2))+  r'\\')
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
            
            
        f.write(r'& Fixed  &'+str(int(out_it[1]))+' & '+str(out_time[1])+
                        ' & '+str(round(out_Fval[1],2))+   r'\\')
        
        f.write('\n')
        
        
        f.write(r'Adaptive & Decreasing  &'+str(int(out_it[3]))+' & '+str(out_time[3])+
                        ' & '+str(round(out_Fval[3],2))+   r'\\')
        
        f.write('\n')
        
        f.write(r'& Adaptive  &'+str(int(out_it[5]))+' & '+str(out_time[5])+
                        ' & '+str(round(out_Fval[5],2))+  r'\\')
        
        f.write('\n')
        
        f.write("\\hline")
            
        f.write('\n')
        f.write("\\end{tblr}")
    