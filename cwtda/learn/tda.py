import os

import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
from gudhi import representations

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

def point_cloud_to_pl(point_cloud, sample_range=[0, .6], resolution=100, label='test'):
    # Making folder for saving
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'concept', label), exist_ok=True)

    # Persistance Barcode and Diagram
    rips_complex = gd.RipsComplex(points=point_cloud)
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
    pl = [pl.transform([diag_1])[0][i*pl.resolution: (i+1)*pl.resolution] for i in range(pl.num_landscapes)]

    return pl_x, pl_lst