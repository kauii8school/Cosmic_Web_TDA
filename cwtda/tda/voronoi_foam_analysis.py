import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from gudhi import representations

from .. import voronoi_foams


def load_data(data_folder_name):

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "voronoi_foams",
            "results",
            data_folder_name,
            "beams.pkl",
        ),
        "rb",
    ) as f:
        beams = pickle.load(f)

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "voronoi_foams",
            "results",
            data_folder_name,
            "seeds.pkl",
        ),
        "rb",
    ) as f:
        seeds = pickle.load(f)

    return seeds, beams

def plot_data(data_folder_name):
    load_data(data_folder_name)
    seeds, beams = load_data(data_folder_name)

    voronoi_foams.debug_plot.plot_3d_all(seeds, beams)

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
