import os
import time
import logging

import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
from gudhi import representations
from sklearn.neighbors import NearestNeighbors


def plot_persistance_landscape(pl_x, pl, ax=None,  show=False, **plt_kwargs):
    """ 
    Parameters
    -------------------
    pl_x, : np.array 
        x values (parameter value) of persistance landscape
    
    pl_lst : list
        List of np.arrays where each np.array is a persistance landscape
    
    show=False : bool 
        If you want to plot the persistance landscape and not just return the matplotlib axes
    
    Returns
    -------------------
    mean_pl : list
        Average persistance landscape at epoch
    """

    if ax is None:
        ax = plt.gca() 

    ax.set_xlabel('Parameter Values')
    ax.set_ylabel('Persistance')
    for landscape in pl:
        ax.plot(pl_x, landscape, **plt_kwargs)

    if show:
        plt.show()

    return(ax)

def get_average_persistance_landscape(epoch, pl_x_lst, pl_lst):
    """ 
    Parameters
    -------------------
    epoch : int
        Epoch to get the average persistance landscape of 

    pl_x_lst : list
        List of x persistance landscape values 

    pl_lst : list
        List of np.arrays of persistance landscape values
    
    Returns
    -------------------
    mean_pl : list
        Average persistance landscape at epoch
    """
    epoch_pl_lst = [pl[1] for pl in pl_lst if pl[0] == epoch]
    stacked_pl = [np.stack([pl[j] for pl in epoch_pl_lst]) for j in range(len(epoch_pl_lst[0]))]
    mean_pl = [np.mean(stack_pl, axis=0) for stack_pl in stacked_pl]

    return mean_pl

def test_annulus():
    """
    Testing sci-kit tda on annulus to make sure everything works the same as in R. Output is 4 plots
    the sampled points, a persistance diagram, a persistance bardcode, and persistance landscape
    """

    # Making axes for us plot on
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Generating annulus
    inner_radius = 2
    outer_radius = 5
    theta = np.random.uniform(0, 2 * np.pi, 100)
    r = np.sqrt(np.random.uniform(inner_radius ** 2, outer_radius ** 2, 100))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.c_[x, y]
    axs[0, 0].scatter(x, y)

    # Persistance Barcode and Diagram
    rips_complex = gd.RipsComplex(points=data)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence(min_persistence=0)

    gd.plot_persistence_diagram(diag, axes=axs[1, 0], legend=True)

    gd.plot_persistence_barcode(diag, axes=axs[1, 1], legend=True)

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


    for i in range(pl.num_landscapes):
        axs[0, 1].plot(pl_x, pl_output[i])

    plt.show()

def point_cloud_to_pl(point_cloud, sample_range=[0, .6], resolution=100):
    """ 
    Parameters
    -------------------
    point_cloud : np.array
        NxM array where N is the number of points and M is the dimension

    sample_range ([float, float]) – minimum and maximum of all piecewise-linear function domains, of the form [x_min, x_max] 
    (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, 
    it can be computed from the persistence diagrams with the fit() method.
    
    resolution : int
        Number of sample for all piecewise-linear functions (default 100).

    
    Returns
    -------------------
    pl_x_1 : np.array
        samples on interval for parameters in persistance landscape for Homology in degree 1
    
    pl_x_2 : np.array
        samples on interval for parameters in persistance landscape for Homology in degree 2

    pl_1_lst : list of np.array
        Piecewise linear persistance landscapes in degree 1
    
    pl_2_lst : list of np.array
        Piecewise linear persistance landscapes in degree 2
    """
    tstart = time.time()
    # Persistance Barcode and Diagram
    logging.info("Calculating Sparse Rips Complex")
    rips_complex = gd.RipsComplex(points=point_cloud, sparse=0.3)
    logging.info("Calculating Simplex Tree")
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    logging.info("Calculating Persistence Diagram")
    diag = simplex_tree.persistence(min_persistence=0)

    # Persistance Landscape
    diag_0 = np.array([np.array(i[1]) for i in diag if i[0] == 0])
    diag_1 = np.array([np.array(i[1]) for i in diag if i[0] == 1])
    diag_2 = np.array([np.array(i[1]) for i in diag if i[0] == 2])

    # Set a high value for the max death time can't be infinite though (max + average)
    max_filtration_1 = (
        np.ma.masked_invalid(diag_1[:, 1]).max()
        + np.ma.masked_invalid(diag_1[:, 1]).mean()
    )
    diag_1[np.where(diag_1 == np.inf)] = max_filtration_1

    if diag_2 == []:
        max_filtration_2 = (
            np.ma.masked_invalid(diag_2[:, 1]).max()
            + np.ma.masked_invalid(diag_2[:, 1]).mean()
        )
    else:
        max_filtration_2 = 0
    diag_2[np.where(diag_2 == np.inf)] = max_filtration_2

    logging.info("Calculating Persistence Landscape")
    pl_1 = representations.vector_methods.Landscape(sample_range=sample_range, resolution=resolution).fit([diag_1])
    pl_x_1 = np.linspace(pl_1.sample_range[0], pl_1.sample_range[1], pl_1.resolution)
    pl_1_lst = [pl_1.transform([diag_1])[0][i*pl_1.resolution: (i+1)*pl_1.resolution] for i in range(pl_1.num_landscapes)]

    pl_2 = representations.vector_methods.Landscape(sample_range=sample_range, resolution=resolution).fit([diag_2])
    pl_x_2 = np.linspace(pl_2.sample_range[0], pl_2.sample_range[1], pl_2.resolution)
    pl_2_lst = [pl_2.transform([diag_2])[0][i*pl_2.resolution: (i+1)*pl_2.resolution] for i in range(pl_2.num_landscapes)]

    logging.info(f"Completed... Time elapsed {round(time.time() - tstart)}s")

    return pl_x_1, pl_x_2, pl_1_lst, pl_2_lst

def knn_denoise(point_cloud, n_neighbors, nbs_frac=.5):
    """ 
    Parameters
    -------------------
    point_cloud : np.array
        NxM array where N is the number of points and M is the dimension

    n_neighbors : int
        number of neighbors to use for nearest distance calculation to remove points 

    nbs_frac : float
        Fraction that is multiplied by n_neighbors to cut off points that are too close to to
        each other. A high number is suggested for earlier simulations


    Returns
    -------------------
    point_cloud : np.array
        (N-L)xM point cloud of data where L is the number of points removed. 
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)

    sum_dist_arr = np.sum(distances, axis=1)
    avg_dist = np.average(sum_dist_arr)
    rem_idxs = np.where(((n_neighbors * nbs_amt) <= sum_dist_arr) & (sum_dist_arr <= avg_dist))
    masked_pos = point_cloud[rem_idxs]
    
    return masked_pos