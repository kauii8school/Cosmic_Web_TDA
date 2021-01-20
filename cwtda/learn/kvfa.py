import pickle
import os

import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
from gudhi import representations

from .. import kinematic_voronoi_foams as kvf 

def load_data(data_folder_name):
    """ 
    Loading data from kinematic voronoi foam simulation

    Parameters 
    ---------------
    data_folder_name : str
        data folder in cwtda/kinematic_voronoi_foams/data that we want to get data from
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "kinematic_voronoi_foams", "data", data_folder_name, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    return data 

def plot_persistance(data_folder_name, sample_range=[0, .6], resolution=100):
    """ 
    Will plot persistance landscapes corresponding to data_folder_name. Note this does not used precomputed
    persistance landscapes, barcodes or diagrams so it may take a while to compute.  

    Parameters 
    ---------------
    data_folder_name : str
        data folder in cwtda/kinematic_voronoi_foams/data that we want to get data from
    """

    simulation_data = load_data(data_folder_name)
    
    # Making folder for saving
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder_name, 'frames'), exist_ok=True)

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

        pl = representations.vector_methods.Landscape(sample_range=sample_range, resolution=resolution).fit([diag_1])
        pl_x = np.linspace(pl.sample_range[0], pl.sample_range[1], pl.resolution)
        pl_output = [pl.transform([diag_1])[0][i*pl.resolution: (i+1)*pl.resolution] for i in range(pl.num_landscapes)]

        axs[0, 1].set_ylim([0, .1])
        axs[0, 1].set_xlabel('Parameter Values')
        axs[0, 1].set_ylabel('Persistance')
        for i in range(pl.num_landscapes):
            axs[0, 1].plot(pl_x, pl_output[i])

        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder_name, 'frames', f'{tc_pair[0]:03d}'))
        plt.close()

def get_persistance_landscapes(data_folder_name, sample_range=[0, .6], resolution=100):
    simulation_data = load_data(data_folder_name)
    
    # Making folder for saving
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder_name), exist_ok=True)

    pl_lst = []
    for tc_pair in simulation_data:
        data = tc_pair[1]

        # Persistance Barcode and Diagram
        rips_complex = gd.RipsComplex(points=data)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        diag = simplex_tree.persistence(min_persistence=0)

        # Persistance Landscape
        diag_0 = np.array([np.array(i[1]) for i in diag if i[0] == 0])
        diag_1 = np.array([np.array(i[1]) for i in diag if i[0] == 1])

        # Set a high value for the max death time can't be infinite though (max + average)
        max_filtration_1 = (
            np.ma.masked_invalid(diag_1[:, 1]).max()
            + np.ma.masked_invalid(diag_1[:, 1]).mean()
        )
        diag_1[np.where(diag_1 == np.inf)] = max_filtration_1

        pl = representations.vector_methods.Landscape(sample_range=sample_range, resolution=resolution).fit([diag_1])
        pl_x = np.linspace(pl.sample_range[0], pl.sample_range[1], pl.resolution)
        pl_output = [pl.transform([diag_1])[0][i*pl.resolution: (i+1)*pl.resolution] for i in range(pl.num_landscapes)]
        pl_lst.append((tc_pair[0], pl_output))

    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder_name, 'pl.pkl'), 'wb') as f:
        pickle.dump(pl_lst, f)

    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder_name, 'pl_x.pkl'), 'wb') as f:
        pickle.dump(pl_x, f)

def load_persistance_landscapes(data_folder_lst=None):
    """ 
    Will load and return persistance landscapes. If no data folder name is specified will load
    all persistance landscapes. 

    Parameters 
    ---------------
    data_folder_lst=None : list
        List of data folders we want to load persistance landscapes from 

    Returns 
    --------------
    total_pl_x_lst : list 
        List of x points where the persistance landscapes are plotted on 

    total_pl_lst : list
        List of tuples where the first is the label and the second is the 
        landscape associated with said label. Note here the labels are 
        epochs (MESED UP I THINK?)
    """

    if data_folder_lst == None: 
        data_folder_lst = os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa'))

    total_pl_x_lst, total_pl_lst = [], []
    for data_folder in data_folder_lst:
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder, 'pl.pkl'), 'rb') as f:
            pl_lst = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'kvfa', data_folder, 'pl_x.pkl'), 'rb') as f:
            pl_x = pickle.load(f)

        total_pl_x_lst.append(pl_x)
        total_pl_lst.extend(pl_lst)

    return total_pl_x_lst, total_pl_lst

