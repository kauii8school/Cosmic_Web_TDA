#%%
import os 
import logging
import datetime
import sys
import time
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from cwtda.learn import kvfa 
from cwtda.learn import tda
from cwtda.concept import dataio, plotting
from cwtda import config

# kvfa.plot_persistance('1')

# Logging
# logging.basicConfig(
#     level=logging.INFO,
#     filemode='a',
#     filename=os.path.join(config.CWTDA_DIR, '..', 'logfiles', f'logfile_{datetime.datetime.now()}.log'),
#     datefmt='%H:%M:%S',
# )
# root = logging.getLogger()
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# root.addHandler(handler)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%%
# Where the concept simulation files are stored 
sim_path = "/home/n/Documents/Research/etc_codes/CONCEPT/concept/output/simulation_1"
concept_sim_snap_lst = dataio.load_data(sim_path, "BaseSimulation")

for i, sim_snap in enumerate(concept_sim_snap_lst):
    sim_snap.cut_cube()
    sim_snap.knn_denoise()
    sim_snap.create_pl(max_pts=3000)
    sim_snap.save()

exit()
#%%
# Loading saved files for training
sim = '.20210122190504519'
concept_sim_snap_lst = dataio.load_data(f"/home/n/Documents/Research/Cosmic_Web_TDA/cwtda/data/concept/{sim}", "AnalyzedSimulation")
concept_sim_snap_lst = sorted(concept_sim_snap_lst, key=lambda x: x.time_a)
# for i, sim_snap in enumerate(concept_sim_snap_lst):
#     try:
#         plotting.visual_data_pl(sim_snap)
#         os.makedirs(f"/home/n/Documents/Research/Cosmic_Web_TDA/Plots/{sim}/visual_pl", exist_ok=True)
#         plt.savefig(f"/home/n/Documents/Research/Cosmic_Web_TDA/Plots/{sim}/visual_pl/{i}.png")
#         plt.close()
#     except:
#         pass

# %%
from scipy.stats.kde import gaussian_kde
sim_snap = concept_sim_snap_lst[0]
x, y = sim_snap.pos[:, 0], sim_snap.pos[:, 1]

k = gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=(5,15))
ax0 = fig.add_subplot(313)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)

ax0.scatter(x, y, s=.1)

# alpha=0.5 will make the plots semitransparent
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(y.min(), y.max())
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(y.min(), y.max())
plt.savefig("/home/n/Documents/Research/Cosmic_Web_TDA/Plots/.20210122190504519/density_map_last.png")

# %%
lb, ub = 50, 90
sim_snap = concept_sim_snap_lst[-1]
cubed_pos = np.array([pt for pt in sim_snap.pos if (lb < pt[0] < ub and lb < pt[1] < ub and lb < pt[2] < ub)])

print(cubed_pos.shape)
plt.scatter(cubed_pos[:, 0], cubed_pos[:, 1], s=.1)
# %%
# Using KNN to denoise
from sklearn.neighbors import NearestNeighbors

tstart = time.time()
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(cubed_pos)
distances, indices = nbrs.kneighbors(cubed_pos)
print(f"ran in {round(time.time() - tstart)} s")

# %%
sum_dist_arr = np.sum(distances, axis=1)
avg_dist = np.average(sum_dist_arr)
rem_idxs = np.where((5 <= sum_dist_arr) & (sum_dist_arr <= avg_dist))
masked_pos = cubed_pos[rem_idxs]
print(masked_pos.shape)
plt.scatter(masked_pos[:, 0], masked_pos[:, 1], s=.1)

# %%
tstart = time.time()
pl_x_1, pl_x_2, pl_1_lst, pl_2_lst = tda.point_cloud_to_pl(masked_pos)

print(f"ran in {round(time.time() - tstart)} s")

# %%
for landscape in pl_1_lst:
    plt.plot(pl_x_1, landscape)

# %%
for landscape in pl_2_lst:
    plt.plot(pl_x_2, landscape)
# %%
# import re 
# sim_path = "/home/n/Documents/Research/Cosmic_Web_TDA/cwtda/data/concept/.20210122190504519"
# for fname in os.listdir(sim_path):
#     filepath = os.path.join(sim_path, fname)
#     sim_snap = dataio.ConceptSimulationSnap.load(filepath)
#     sim_snap.param_class = "simulation_2"
#     concept_sim_dir = os.path.join(config.CONCEPT_INSTALL_DIR, "output", sim_snap.param_class)
#     pic_2D_filepath = os.path.join(concept_sim_dir, f"render2D_a={sim_snap.time_a}.png")
#     pic_3D_filepath = os.path.join(concept_sim_dir, f"render3D_a={sim_snap.time_a}.png")
#     print(pic_2D_filepath, pic_3D_filepath, concept_sim_dir)
#     sim_snap.pic_2D_filepath = pic_2D_filepath
#     sim_snap.pic_3D_filepath = pic_3D_filepath
#     sim_snap.concept_sim_dir = concept_sim_dir
#     sim_snap.save(filepath='base')

