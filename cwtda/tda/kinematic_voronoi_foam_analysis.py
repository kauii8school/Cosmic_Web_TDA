import pickle
import os

import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
from gudhi import representations

from .. import kinematic_voronoi_foams as kvf 

def load_data(data_folder_name):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "kinematic_voronoi_foams", "data", data_folder_name, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    return data 

def plot_persistance(data_folder_name):

    simulation_data = load_data(data_folder_name)
    
    # Making folder for saving
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', data_folder_name, 'frames'), exist_ok=True)

    for tc_pair in simulation_data:
        # Making axes for us plot on
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        data = tc_pair[1]
 
        axs[0, 0].scatter(data[:, 0], data[:, 1], c='#FFA500', marker='x', label='galaxy')
        axs[0, 0].set_xlim([0, 1])
        axs[0, 0].set_ylim([0, 1])
        axs[0, 0].set_xlabel('[100 Mpc]')
        axs[0, 0].set_ylabel('[100 Mpc]')
        axs[0, 0].legend(loc='upper left')

        # Persistance Barcode and Diagram
        rips_complex = gd.RipsComplex(points=data)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        diag = simplex_tree.persistence(min_persistence=0)

        gd.plot_persistence_diagram(diag, axes=axs[1, 0], legend=True)
        axs[1,0].set_ylim([0, .34])

        gd.plot_persistence_barcode(diag, axes=axs[1, 1], legend=True)
        axs[1, 1].set_xlim([0, .34])

        # Persistance Landscape
        diag_0 = np.array([np.array(i[1]) for i in diag if i[0] == 0])
        diag_1 = np.array([np.array(i[1]) for i in diag if i[0] == 1])

        # Set a high value for the max death time can't be infinite though (max + average)
        max_filtration_1 = (
            np.ma.masked_invalid(diag_1[:, 1]).max()
            + np.ma.masked_invalid(diag_1[:, 1]).mean()
        )
        diag_1[np.where(diag_1 == np.inf)] = max_filtration_1

        max_filtration_0 = (
            np.ma.masked_invalid(diag_0[:, 1]).max()
            + np.ma.masked_invalid(diag_0[:, 1]).mean()
        )
        diag_0[np.where(diag_0 == np.inf)] = max_filtration_0

        pl = representations.vector_methods.Landscape().fit([diag_1])
        pl_x = np.linspace(pl.sample_range[0], pl.sample_range[1], pl.resolution)
        pl_output = [pl.transform([diag_1])[0][i*pl.resolution: (i+1)*pl.resolution] for i in range(pl.num_landscapes)]


        axs[0, 1].set_ylim([0, .1])
        for i in range(pl.num_landscapes):
            axs[0, 1].plot(pl_x, pl_output[i])

        plt.savefig(os.path.join(os.path.dirname(__file__), 'data', data_folder_name, 'frames', f'{tc_pair[0]:03d}'))
        plt.close()


def get_persistance_landscapes(data_folder_name):

    pass
