import gp_emulator
from BSM_soil import BSM
import numpy as np

f = np.load('BSM_paras.npz')
GSV = f.f.GSV
nw  = f.f.nw
kw  = f.f.kw
BSM_paras = GSV, nw, kw
    
import gp_emulator 



paras = ["n", "cab", "car", "cbrown", "cw", "cm", "lai", "ala", 'sza', 'vza', 'raa', 'B', 'lat', 'lon', 'SMp']
pmins =     [0.5,   0,     0,        0., 0.00,  0.00,    0.,   20.,    0,      0,    0,    0,    10,    22,     2] 
pmaxs =     [3.5, 120,   25,         1., 0.08,  0.04,    8.,   80.,   80,     15,  360,  1.5,    80,   130,   100]


paras = ['B', 'lat', 'lon', 'SMp']
pmins = [  0,    10,    22,     2]
pmaxs = [1.5,    80,   130,   100]

num_samp = 1000000
sample, distributions = gp_emulator.create_training_set(paras, pmins, pmaxs, n_train=int(num_samp * 2))

soil_specs = []
for i in range(len(sample)):
    B, lat, lon, SMp = sample[i][-4:]
    soil_ref = BSM(B, lat, lon, SMp, BSM_paras)
    soil_ref = np.concatenate([soil_ref.ravel(), np.zeros(100)])
    soil_specs.append(soil_ref) 
#     if np.all((soil_ref >= 0) & soil_ref <=1):
#         with open('/home/users/marcyin/marcyin/prosail/soil_sample.txt', 'a') as f:
#             para = [B, lat, lon, SMp] + soil_ref.tolist() 
#             para_str = ','.join(map(str, para))
#             f.write(para_str + '\n')
            
soil_specs = np.array(soil_specs)
mask = np.all((soil_specs >= 0) & (soil_specs <= 1), axis=1)
soil_specs = soil_specs[mask][:num_samp]
sample = sample[mask][:num_samp]

np.savez('/home/users/marcyin/marcyin/prosail/soil_samples.npz', sample=sample, soil_specs=soil_specs)