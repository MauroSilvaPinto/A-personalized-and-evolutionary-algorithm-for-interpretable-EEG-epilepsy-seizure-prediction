'''
 translated to python by Mauro Pinto, adapted from
 M. Quyen Matlab Version.
 
 The random predictor, from Schelter et al.

'''


def f_RandPredictd(n_SzTotal, s_FPR, d, s_SOP, alpha):
    import numpy as np
    import math
    
    from scipy.special import comb
    #[s_kmax, p_value, v_SumSignif] = f_RandPredictd(5,0.07,1,20/60)
    # Random predictor with d free independent parameters

    v_PBinom = np.zeros(n_SzTotal)
    s_kmax = 0
    
    
    # o +1, -1 tem a ver com no matlab a iteracao comeca em 1, aqui em 0 :)
    for seizure_i in range(0,n_SzTotal):
        v_Binom=comb(n_SzTotal,seizure_i+1)
        #s_PPoi=1-math.exp(-s_FPR*s_SOP)
        s_PPoi=s_FPR*s_SOP
        v_PBinom[seizure_i]=v_Binom*s_PPoi**(seizure_i+1)*((1-s_PPoi)**(n_SzTotal-seizure_i-1))
        
    #p_value=1-(1-np.cumsum(np.flip(v_PBinom)))**d
    v_SumSignif=1-(1-np.cumsum(np.flip(v_PBinom)))**d>alpha
    s_kmax=np.count_nonzero(v_SumSignif)/n_SzTotal
    
    return s_kmax

    

