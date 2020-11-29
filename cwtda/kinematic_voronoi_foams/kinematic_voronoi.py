import sys
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial
from matplotlib import animation
import scipy.stats

global eps
eps = sys.float_info.epsilon

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))

def voronoi(towers, bounding_box):
    """ https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells """
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)
    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor

def centroid_region(vertices):
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])

def plot_voronoi_2d(voro, galaxies, file_name=None, plot_voronoi=True):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    if plot_voronoi:
        # Plot initial points
        ax.plot(voro['vor'].filtered_points[:, 0], voro['vor'].filtered_points[:, 1], 'b.', label='voronoi center')
        # Plot ridges points
        for region in voro['vor'].filtered_regions:
            vertices = voro['vor'].vertices[region, :]
            ax.plot(vertices[:, 0], vertices[:, 1], 'go', label='vertex')
        # Plot ridges
        for region in voro['vor'].filtered_regions:
            vertices = voro['vor'].vertices[region + [region[0]], :]
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-', label='ridge')

        # Plotting ridge midpoints
        x, y = voro['ridge_centers'][:, 0], voro['ridge_centers'][:, 1]
        ax.scatter(x, y, c='r', label='ridge midpoint')
        
    ax.scatter(galaxies[:, 0], galaxies[:, 1], c='#FFA500', marker='x', label='galaxy')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    if file_name != None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.close()

def generate_time_series_voronoi_2d(num_nuclei=5, mean_density=5, edge_length=1, simulation_name='test', plot=False):
    """ 

    Implementation of https://academic.oup.com/mnras/article/380/2/551/1010947#91677586 appendix C

    Parameters
    -----------------
    num_nuclei : int
        number of expansion centers to create within the volume. 

    mean_density : int 
        intensity (ie mean density) of the Poisson process

    edge_length : int 
        edge length of the volume. The total volume is just the edge length in cubed. Note this is in in Mpc

    """

    
    bounding_box = np.array([0., edge_length, 0., edge_length]) # [x_min, x_max, y_min, y_max]

    nuclei = np.random.rand(num_nuclei, 2) * edge_length



    #Simulation window parameters
    xMin=0;xMax=edge_length;
    yMin=0;yMax=edge_length;
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta;

    #Point process parameters
    lambda0=100; #intensity (ie mean density) of the Poisson process
    
    #Simulate Poisson point process
    numbPoints = scipy.stats.poisson( mean_density*areaTotal ).rvs()#Poisson number of points
    xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
    yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points
    galaxies = np.c_[xx, yy]


    # Getting ridge points and defining voronoi tesselation
    vor = voronoi(nuclei, bounding_box)
    voro = {'vor':vor}
    ridge_centers = np.array([])
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        for v0, v1 in zip(vertices[:-1], vertices[1:]):
            ridge_center = np.array([(v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2])
            ridge_centers = np.append(ridge_centers, ridge_center)
    ridge_centers = ridge_centers.reshape(-1, 2)

    # Iterating through the regions and appending points
    arr = np.array([])
    for region in vor.filtered_regions:
        arr = np.append(arr, vor.vertices[region, :])
    arr = arr.reshape(-1, 2)

    voro['ridge_centers'] = ridge_centers
    voro['centers'] = vor.filtered_points
    voro['vertices'] = arr 

    # Creating folder to save data
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', f'{simulation_name}', 'frames_vor'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data', f'{simulation_name}', 'frames_gal'), exist_ok=True)

    # For each galaxy calculating the vector direction it should move
    # Moving radially away from the voronoi center 
    orig_galaxies = galaxies

    def void_expansion(t, tp):
        if tp == 'center':
            return 6
        elif tp == 'vertex':
            return 2

    galaxy_locations = []
    for t in range(25):
        get_closest_center = lambda x: min(voro['centers'], key=lambda y: np.linalg.norm(x - y))
        closest_centers = np.array([get_closest_center(galaxy) for galaxy in galaxies])
        uv_center = (galaxies - closest_centers) / 100 

        get_closest_vert = lambda x: min(voro['vertices'], key=lambda y: np.linalg.norm(x - y))

        global_mask = np.array([(True, True) for galaxy in galaxies])
        if t > 0:
            # Copying to see if it got changed
            closest_centers_copy = closest_centers.copy()

            mask = closest_centers_copy == closest_centers
            # If it goes past the "bounds"
            bdd_mask = np.array([(False, False) if (galaxy[0] > edge_length or galaxy[1] > edge_length or galaxy[0] < 0 or galaxy[1] < 0) else (True, True) for galaxy in galaxies])
            
            mask = np.logical_and(mask, bdd_mask)
            global_mask = np.logical_and(mask, global_mask)

            galaxies[global_mask] = galaxies[global_mask] + void_expansion(t, 'center')*uv_center[global_mask]

            closest_ridge = np.array([get_closest_vert(galaxy) for galaxy in galaxies])
            uv_vertex = (closest_ridge - galaxies) / 100

            closest_vert = np.array([get_closest_vert(galaxy) for galaxy in galaxies])
            uv_vert = (closest_vert - galaxies) / 100 
            galaxies[np.logical_not(global_mask)] = galaxies[np.logical_not(global_mask)] + void_expansion(t, 'vertex')*uv_vert[np.logical_not(global_mask)]

            if plot:
                plot_voronoi_2d(voro, galaxies, os.path.join(os.path.dirname(__file__), 'data', f'{simulation_name}', 'frames_vor', f"{t:03d}"), plot_voronoi=True)
                plot_voronoi_2d(voro, galaxies, os.path.join(os.path.dirname(__file__), 'data', f'{simulation_name}', 'frames_gal', f"{t:03d}"), plot_voronoi=False)
                
            galaxy_locations.append((t, galaxies.copy()))


    with open(os.path.join(os.path.dirname(__file__), 'data', f'{simulation_name}', 'data.pkl'), 'wb') as f:
        pickle.dump(galaxy_locations, f)
