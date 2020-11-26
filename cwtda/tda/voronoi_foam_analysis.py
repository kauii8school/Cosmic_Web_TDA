import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from gudhi import representations

from .. import voronoi_foams


def load_data(data_folder_name):

    print(data_folder_name)

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "voronoi-foams",
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
            "voronoi-foams",
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
    """ Testing sci-kit tda on annulus to make sure everything works the same as in R """

    # Making axes for us plot on
    fig, axs = plt.subplots(2, 2)

    # Generating annulus
    theta = np.random.uniform(0, 2 * np.pi, 100)
    r = np.sqrt(np.random.uniform(1 ** 2, 2 ** 2, 100))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.c_[x, y]
    axs[0, 0].scatter(x, y)

    # Persistance Barcode and Diagram
    rips_complex = gd.RipsComplex(points=data, max_edge_length=0.7)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence(min_persistence=0.1)

    axs[0, 1] = gd.plot_persistence_diagram(diag, axes=axs[1, 0])

    gd.plot_persistence_barcode(diag, axes=axs[1, 1])

    # Persistance Landscape
    diag_0 = np.array([i[1] for i in diag if i[0] == 0])
    diag_1 = np.array([i[1] for i in diag if i[0] == 1])
    pl_constructor = [diag_0, diag_1]
    pl = representations.vector_methods.Landscape().fit_transform(pl_constructor)
