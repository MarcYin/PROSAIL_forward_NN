from BSM_soil import BSM
from scipy.optimize import minimize
import prosail
import pandas as pd
import datetime
import numpy as np
import gp_emulator 

f = np.load('/home/users/marcyin/nceo_ard/prosail/soil_samples.npz')
soilSample = f.f.sample
soilRef = f.f.soil_specs

numSamp = len(soilSample)

paraNames = ["n", "cab", "car", "cbrown", "cw", "cm", "lai", "ala", 'sza', 'vza', 'raa']
pmins =     [1,   0,     0,        0.,   0.00,  0.00,     0.,   20.,    0,      0,    0] 
pmaxs =     [3.5, 120,   25,       1.,   0.08,  0.04,    12.,   80.,   80,     15,  360]
sample, distributions = gp_emulator.create_training_set(paraNames, pmins, pmaxs, n_train=numSamp)

simu_refs = []
paras = []
for i in range(numSamp):
    B, lat, lon, SMp = soilSample[i]
    soil_ref = soilRef[i]
    N, cab, car, cbrown, cw, cm, lai, ala, sza, vza, raa = sample[i]
    
    simu_ref = prosail.run_prosail(N, cab, car, cbrown, cw, 
                                    cm, lai, ala, 
                                    0.01, sza, vza, raa,
                                    prospect_version='D',typelidf = 2,
                                    rsoil0 = soil_ref
                                    ) 
    para = [N, cab, car, cbrown, cw, cm, lai, ala, sza, vza, raa, B, lat, lon, SMp]
    
#     with open('/home/users/marcyin/marcyin/prosail/prosail_sample.txt', 'a') as f:
#         para = [N, cab, car, cbrown, cw, cm, lai, ala, sza, vza, raa, B, lat, lon, SMp] + simu_ref.ravel().tolist() 
#         para_str = ','.join(map(str, para))
#         f.write(para_str + '\n')
    simu_refs.append(simu_ref.ravel())
    paras.append(para)
np.savez('/home/users/marcyin/nceo_ard/prosail_refs.npz', paras = paras, simu_refs = simu_refs, pmins = pmins, pmaxs = pmaxs, paraNames = paraNames)