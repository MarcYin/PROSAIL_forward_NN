import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras         
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

import pylab as plt


ex = pd.ExcelFile('/home/users/marcyin/prosail/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xlsx')
s2a = ex.parse('Spectral Responses (S2A)')
s2b = ex.parse('Spectral Responses (S2B)')

ind = int(int(sys.argv[1])-1)

if ind == 0:
    rsr = s2a
    sat = 's2a'
elif ind == 1:
    rsr = s2a
    sat = 's2b'
print(sat)

f = np.load('/home/users/marcyin/nceo_ard/prosail_refs.npz')
samples = f.f.paras
simu_refs = f.f.simu_refs

paras = ['B', 'lat', 'lon', 'SMp']
pmins = [  0,    10,    22,     2]
pmaxs = [1.5,    80,   130,   100]

"n", "exp(-cab/100)", "exp(-car/100)", "cbrown", "exp(-50*cw)", "exp(-50*cm)", "exp(-lai/2)", "ala/90.", "bsoil", "psoil"
samples[:,0] = (samples[:,0] - 1) / 2.5
samples[:,1] = np.exp(-samples[:,1]/100.)
samples[:,2] = np.exp(-samples[:,2]/100.)
samples[:,4] = np.exp(-50.*samples[:,4])
samples[:,5] = np.exp(-50.*samples[:,5])    
samples[:,6] = np.exp(-samples[:,6]/2.)    
samples[:, 7] = np.cos(np.deg2rad(samples[:, 7]))
samples[:,8:10] = np.cos(np.deg2rad(samples[:,8:10]))
samples[:,10] = samples[:,10] /360

samples[:,11] = samples[:,11] / 1.5
samples[:,12] = (samples[:,12] - 10) / 70
samples[:,13] = (samples[:,13] - 22) / (130 - 22)
samples[:,14] = (samples[:,14] - 2 ) / (100 - 2)


def get_s2_spec(simu_ref, b_ind):
    veg_ref = simu_ref[:,:2000].T
    out = []
    for i in b_ind:                                                     
        re1 = np.nansum((veg_ref * rsr.iloc[100:2100, i+1].values[:,None]), axis=0) / \
             np.nansum(rsr.iloc[100:2100, i+1].values) 
        out.append(re1) 
    out = np.array(out)
    return out

b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
spec_with_soil = get_s2_spec(simu_refs, b_ind)

del simu_refs

mask = np.all(spec_with_soil.T >= 0, axis=1) & np.all(spec_with_soil.T <= 1, axis=1)

xx = samples[mask]
target = spec_with_soil.T[mask]

nodes = 128
inputs = layers.Input(shape=(xx.shape[1],))    
x = layers.Dense(nodes, activation='relu')(inputs)
x = layers.Dense(nodes, activation='relu')(x)
x = layers.Dense(nodes, activation='relu')(x)
output = layers.Dense(target.shape[1])(x) 

model = tf.keras.Model(inputs=inputs, outputs=output)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'foward_prosail_%s_v2.h5'%sat, save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=100, min_lr=0
    ),
#     keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, verbose=1, min_delta=1e-16, restore_best_weights=True),
]
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)

# Split the data
x_train, x_valid, y_train, y_valid = train_test_split(xx, target, test_size=0.1, shuffle= True)

epochs = 2000
batch_size = 32

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data = (x_valid, y_valid), 
    verbose=1,
)

ret = model.predict(x_valid)
fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (20, 10))
axs = axs.ravel()
for j in range(target.shape[1]):

    minv = np.array([y_valid[:, j].ravel(), ret[:, j].ravel()]).min() * 0.8
    maxv = np.array([y_valid[:, j].ravel(), ret[:, j].ravel()]).max() * 1.2

    rmse = np.sqrt(np.mean((y_valid[:, j].ravel()- ret[:, j].ravel())**2))
    axs[j].plot(y_valid[:, j].ravel(), ret[:, j].ravel(), 'ro', ms=3, mew=1, mfc='none', alpha=0.1)

    axs[j].text(0.2 * (maxv - minv) + minv, 0.8 * (maxv - minv) + minv, 'RMSE: %.05f'%rmse)
    axs[j].plot([0, 100], [0,100])
    axs[j].set_xlim(minv, maxv)
    axs[j].set_ylim(minv, maxv)
plt.savefig('./prosail_%s_v2.png'%sat, dpi=300)