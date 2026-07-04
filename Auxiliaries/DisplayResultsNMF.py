# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:05:29 2025

Authors: Pedro Pérez-Aros, David Torregrosa-Belén

Code associated to the paper: Randomized block proximal method with locally
Lipschitz continuous gradient
"""

import numpy as np

def generate_table(fname, vranks, out_it, out_time, out_Fval, out_PSNR, out_feval):
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
        
        table_format = r"\begin{tblr}{width = \linewidth,colspec = {QQS[table-format=6,detect-weight,  mode=text]S[table-format=7,detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=2.2, detect-weight,  mode=text]},}"
        
        f.write(table_format)
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        
        f.write(r"{{{Rank}}} & Method & {{{\# Iterations}}} & {{{\# eval. $F$}}} & {{{Time (s.)}}} & {{{$F(U^{out},V^{out})$}}} & {{{PSNR}}} \\")
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        for dd in range(len(vranks)):
            
            
            f.write(r'& RNBPG &'+str(int(out_it[dd,0]))+' & '+str(int(out_feval[dd,0]))+' & '+str(out_time[dd,0])+
                            ' & '+str(round(out_Fval[dd,0],2))+ '&' +str(round(out_PSNR[dd,0],2))+  r'\\')
            
            f.write('\n')
            
            f.write(str(vranks[dd]))

            
            f.write(r'& ARBPG &'+str(int(out_it[dd,1]))+' & '+str(int(out_feval[dd,1]))+' & '+str(out_time[dd,1])+
                ' & '+str(round(out_Fval[dd,1],2))+ '&' +str(round(out_PSNR[dd,1],2))+  r'\\')


            
            f.write('\n')
            
            f.write(r'& ARBPG-B &'+str(int(out_it[dd,2]))+' & '+str(int(out_feval[dd,2]))+' & '+str(out_time[dd,2])+
                            ' & '+str(round(out_Fval[dd,2],2))+ '&' +str(round(out_PSNR[dd,2],2))+  r'\\')
            
            f.write('\n')
            
            f.write("\\hline")
            
        f.write('\n')
        f.write("\\end{tblr}")
        
def generate_table_epochs(fname, vranks, out_it, out_time, out_Fval, out_PSNR):
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
        
        table_format = r"\begin{tblr}{width = \linewidth,colspec = {QQS[table-format=4.1,detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=3.2, detect-weight,  mode=text]S[table-format=2.2, detect-weight,  mode=text]},}"
        
        f.write(table_format)
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        
        f.write(r"{{{Image}}} & Method & {{{\# epochs}}}  & {{{Time (s)}}} & {{{$\varphi(V^{out},W^{out})$}}} & {{{PSNR}}} \\")
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        for dd in range(len(vranks)):
            
            f.write(str(vranks[dd]))
            
            f.write(r'& RNBPG $b=1$ &'+str(int(out_it[dd,0]))+' & '+str(out_time[dd,0])+
                            ' & '+str(round(out_Fval[dd,0],2))+ '&' +str(round(out_PSNR[dd,0],2))+  r'\\')
            
            f.write('\n')
            
           

            
            f.write(r'& ARBPG $b=5$ &'+str(int(out_it[dd,1]))+' & '+str(out_time[dd,1])+
                ' & '+str(round(out_Fval[dd,1],2))+ '&' +str(round(out_PSNR[dd,1],2))+  r'\\')


            
            f.write('\n')
            
            f.write(r'& PALM &'+str(int(out_it[dd,2]))+' & '+str(out_time[dd,2])+
                            ' & '+str(round(out_Fval[dd,2],2))+ '&' +str(round(out_PSNR[dd,2],2))+  r'\\')
            
            f.write('\n')
            
            f.write(r'& SCIKIT &'+str(int(out_it[dd,3]))+' & '+str(out_time[dd,3])+
                            ' & '+str(round(out_Fval[dd,3],2))+ '&' +str(round(out_PSNR[dd,3],2))+  r'\\')
            
            f.write('\n')
            
            f.write("\\hline")
            
        f.write('\n')
        f.write("\\end{tblr}")
        
def generate_table_SymNMF(fname, n_samples, out_it, out_time, out_Fval, out_feval, out_accuracy, out_ARI, out_boost):
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
        
        table_format = r"\begin{tblr}{width = \linewidth,colsep = 3pt,        font = \small,     rows = {abovesep=3pt, belowsep=3pt},colspec = {Q[c, valign=m] Q[l, valign=m] X[c, valign=m, si={table-format=4,   detect-weight, mode=text}]X[c, valign=m, si={table-format=4,   detect-weight, mode=text}]X[c, valign=m, si={table-format=4,   detect-weight, mode=text}]X[c, valign=m, si={table-format=2.2, detect-weight, mode=text}]X[c, valign=m, si={table-format=4.2, detect-weight, mode=text}]X[c, valign=m, si={table-format=2.2, detect-weight, mode=text}]X[c, valign=m, si={table-format=.2,  detect-weight, mode=text}]},}"
        
        f.write(table_format)
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        
        f.write(r"{{{Dim.}}} & Method & {{{\# Iterations}}} & {{{\# eval. $F$}}}  &  {{{\# boosts}}} & {{{Time (s.)}}} & {{{$\varphi(H^{out})$}}} & {{{Clust. acc. (\%)}}} & {{{ARI}}} \\")
        
        f.write('\n')
        
        f.write(r"\hline")
        
        f.write('\n')
        for dd in range(len(n_samples)):
            
            f.write(r" \SetCell[c=2]{c}  " + str(n_samples[dd]))
            
            f.write(r'& ARBPG &'+str(int(out_it[dd,0]))+' & '+str(int(out_feval[dd,0]))+' & '+'0'+'&'+str(out_time[dd,0])+
                            ' & '+str(round(out_Fval[dd,0],2))+ '&' +str(round(out_accuracy[dd,0],2))+ '&' +str(out_ARI[dd,0])+ r'\\')
            
            f.write('\n')
            
           

            
            f.write(r'& ARBPG-B &'+str(int(out_it[dd,1]))+' & '+str(int(out_feval[dd,1]))+' & '+str(int(out_boost[dd]))+'&'+str(out_time[dd,1])+
                            ' & '+str(round(out_Fval[dd,1],2))+ '&' +str(round(out_accuracy[dd,1],2))+ '&' +str(out_ARI[dd,1])+ r'\\')


            
          
            
            f.write('\n')
            
            f.write("\\hline")
            
        f.write('\n')
        f.write("\\end{tblr}")
        

def generate_table_SymNMF_sdv(fname, n_samples, out_it, out_feval, out_accuracy, out_ARI, out_boost,
                             output_it_sdv, out_nfeval_sdv, out_accuracy_sdv, out_ARI_sdv, out_nbls_sdv):
    """
    Parameters
    ----------
    fname : name of the output file
    n_samples : array/list with dimensions (Dim.)
    ... (matrices de promedios y de desviaciones típicas _sdv)

    Returns
    -------
    fname.txt: file with latex code for table including mean \pm sdv
    """
    
    # Función auxiliar para formatear la celda como: $ media \pm sdv $
    def fmt_val(mean, sdv, decimals=0):
        fmt = f"{{:.{decimals}f}}"
        return f"${fmt.format(mean)} \\pm {fmt.format(sdv)}$"

    with open(fname, 'w') as f:
        # 1. Modificamos el colspec: cambiamos las columnas 'si' por columnas 'X[c, valign=m]' generales
        table_format = (
            r"\begin{tblr}{"
            r"width = \linewidth, "
            r"colsep = 2.5pt, "  # Reducido sutilmente a 2.5pt para que el "+-" no desborde el margen
            r"font = \small, "
            r"rows = {abovesep=3pt, belowsep=3pt}, "
            r"colspec = {Q[c, valign=m] Q[l, valign=m] X[c, valign=m] X[c, valign=m] X[c, valign=m] X[c, valign=m] X[c, valign=m]},"
            r"}"
        )
        
        f.write(table_format + '\n')
        f.write(r"\hline" + '\n')
        
        # Cabeceras limpias (sin triples llaves obligadas)
        f.write(r"Dim. & Method & {\# It.\newline erations} & {\# eval.\newline $F$} & {\# bo-\newline osts} & {Clust.\newline acc. (\%)} & {ARI} \\" + '\n')
        f.write(r"\hline" + '\n')
        
        for dd in range(len(n_samples)):
            # Primera fila: ARBPG
            f.write(r" \SetCell[c=2]{c}  " + str(n_samples[dd]))
            
            # Formateamos cada celda llamando a la función auxiliar con sus medias y desviaciones correspondientes
            it_cell_0  = fmt_val(out_it[dd, 0], output_it_sdv[dd, 0], decimals=0)
            feval_cell_0 = fmt_val(out_feval[dd, 0], out_nfeval_sdv[dd, 0], decimals=0)
            # ARBPG no tiene boosts, así que es 0 \pm 0
            boost_cell_0 = fmt_val(0, 0, decimals=0) 
            acc_cell_0   = fmt_val(out_accuracy[dd, 0], out_accuracy_sdv[dd, 0], decimals=2)
            ari_cell_0   = fmt_val(out_ARI[dd, 0], out_ARI_sdv[dd, 0], decimals=2)
            
            f.write(f" & ARBPG & {it_cell_0} & {feval_cell_0} & {boost_cell_0} & {acc_cell_0} & {ari_cell_0} \\\\\n")
            
            # Segunda fila: ARBPG-B
            it_cell_1  = fmt_val(out_it[dd, 1], output_it_sdv[dd, 1], decimals=0)
            feval_cell_1 = fmt_val(out_feval[dd, 1], out_nfeval_sdv[dd, 1], decimals=0)
            # Para los boosts de ARBPG-B asumimos que out_boost es la media y out_nbls_sdv es su desviación típica
            boost_cell_1 = fmt_val(out_boost[dd], out_nbls_sdv[dd], decimals=0)
            acc_cell_1   = fmt_val(out_accuracy[dd, 1], out_accuracy_sdv[dd, 1], decimals=2)
            ari_cell_1   = fmt_val(out_ARI[dd, 1], out_ARI_sdv[dd, 1], decimals=2)
            
            f.write(f" & ARBPG-B & {it_cell_1} & {feval_cell_1} & {boost_cell_1} & {acc_cell_1} & {ari_cell_1} \\\\\n")
            
            f.write(r"\hline" + '\n')
            
        f.write(r"\end{tblr}")
        
        
import numpy as np

def generate_table_swimmer(fname, output_epochs_mean, output_time_mean, 
                           output_Fval_mean,  output_epochs_sdv, output_time_sdv, 
                           output_Fval_sdv):
    """
    Parameters
    ----------
    fname : name of the output file
    ... (matrices de promedios y de desviaciones típicas _sdv)

    Returns
    -------
    fname.txt: file with latex code for table including mean \pm sdv in scientific notation
    """
    
    # Función interna con parámetro explícito para forzar notación científica
    def fmt_val(mean, sdv, decimals=2, scientific=False):
        if scientific:
            # Forzamos la extracción del exponente base de la media
            if mean == 0:
                return f"$0 \\pm 0$"
            
            exponent = int(np.floor(np.log10(abs(mean))))
            factor = 10**exponent
            
            # Escalamos los valores con respecto al factor común
            scaled_mean = mean / factor
            scaled_sdv = sdv / factor
            
            return f"$({scaled_mean:.2f} \\pm {scaled_sdv:.2f}) \\cdot 10^{{{exponent}}}$"
        else:
            # Formateo decimal estándar
            return f"${mean:.{decimals}f} \\pm {sdv:.{decimals}f}$"

    with open(fname, 'w') as f:
        table_format = (
            r"\begin{tblr}{"
            r"width = \linewidth, "
            r"colsep = 4pt, "  
            r"font = \small, "
            r"rows = {abovesep=4pt, belowsep=4pt}, "
            r"colspec = {Q[l, valign=m] X[c, valign=m] X[c, valign=m] X[c, valign=m]},"
            r"}"
        )
        
        f.write(table_format + '\n')
        f.write(r"\hline" + '\n')
        
        # Cabeceras
        f.write(r"Method & {\# epochs} & {Time (s)} & {$\varphi(V^{out},W^{out})$} \\" + '\n')
        f.write(r"\hline" + '\n')
        
        # Primera fila: ARBPG (Índice 1)
        epoch_cell_0 = fmt_val(output_epochs_mean[1], output_epochs_sdv[1], decimals=0)
        time_cell_0  = fmt_val(output_time_mean[1], output_time_sdv[1], decimals=2)
        # --- FORZADO AQUÍ CON scientific=True ---
        Fval_cell_0  = fmt_val(output_Fval_mean[1], output_Fval_sdv[1], scientific=True)

        f.write(f" ARBPG & {epoch_cell_0} & {time_cell_0} & {Fval_cell_0} \\\\\n")
        
        # Segunda fila: iPALM (Índice 2)
        epoch_cell_1 = fmt_val(output_epochs_mean[2], output_epochs_sdv[2], decimals=0)
        time_cell_1  = fmt_val(output_time_mean[2], output_time_sdv[2], decimals=2)
        # --- FORZADO AQUÍ CON scientific=True ---
        Fval_cell_1  = fmt_val(output_Fval_mean[2], output_Fval_sdv[2], scientific=True)
        
        f.write(f" iPALM & {epoch_cell_1} & {time_cell_1} & {Fval_cell_1} \\\\\n")
        
        f.write(r"\hline" + '\n')
        f.write(r"\end{tblr}")
        
        
def generate_table_boosted(fname, n_samples, out_lamavg, out_lammax, out_lamavg_sdv, out_lammax_sdv):
    """
    Parameters
    ----------
    fname : name of the output file
    n_samples : array/list with dimensions (Dim.)
    ... (matrices de promedios y de desviaciones típicas _sdv)

    Returns
    -------
    fname.txt: file with latex code for table including mean \pm sdv
    """
    
    # Función auxiliar para formatear la celda como: $ media \pm sdv $
    def fmt_val(mean, sdv, decimals=0):
        fmt = f"{{:.{decimals}f}}"
        return f"${fmt.format(mean)} \\pm {fmt.format(sdv)}$"

    with open(fname, 'w') as f:
        # 1. Modificamos el colspec: cambiamos las columnas 'si' por columnas 'X[c, valign=m]' generales
        table_format = (
            r"\begin{tblr}{"
            r"width = \linewidth, "
            r"colsep = 2.5pt, "  # Reducido sutilmente a 2.5pt para que el "+-" no desborde el margen
            r"font = \small, "
            r"rows = {abovesep=3pt, belowsep=3pt}, "
            r"colspec = {Q[c, valign=m]  X[c, valign=m] X[c, valign=m]},"
            r"}"
        )
        
        f.write(table_format + '\n')
        f.write(r"\hline" + '\n')
        
        # Cabeceras limpias (sin triples llaves obligadas)
        f.write(r"Dim.  & {average $\lambda_k$} & {max $\lambda_k$}  \\" + '\n')
        f.write(r"\hline" + '\n')
        
        for dd in range(len(n_samples)):
            # Primera fila: ARBPG
            f.write( str(n_samples[dd]))
            
            # Formateamos cada celda llamando a la función auxiliar con sus medias y desviaciones correspondientes
            lamavg_cell_0  = fmt_val(out_lamavg[dd], out_lamavg_sdv[dd], decimals=4)
            lammax_cell_0 = fmt_val(out_lammax[dd], out_lammax_sdv[dd], decimals=4)
            # ARBPG no tiene boosts, así que es 0 \pm 0
            
            f.write(f"  & {lamavg_cell_0} & {lammax_cell_0}  \\\\\n")
            

            f.write(r"\hline" + '\n')
            
        f.write(r"\end{tblr}")
        
    
