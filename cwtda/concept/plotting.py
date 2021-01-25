import glob, os, re
import numpy as np
import matplotlib.pyplot as plt

from wand.image import Image
import PIL

def powerspec_plot(sim_path, out_path):
    # Read in data
    P_sims = {}
    for filename in sorted(glob.glob(f'{sim_path}/powerspec*'), key=os.path.getmtime):
        if ".png" in filename:
            continue
        with open(filename, 'r') as f:
            ln = f.readline()
            match = re.search(r't = \d+.\d+ \w+', ln)
            sim = match.group(0)
        k, P_sims[sim], P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)

    # Plot
    plt.figure(figsize=(8, 6))
    for sim, P_sim in P_sims.items():
        plt.loglog(k, P_sim, '-', label=f'simulation {sim}')
    plt.loglog(k, P_lin, 'k--', label='linear', linewidth=1, zorder=np.inf)
    plt.xlim(k[0], k[-1])
    plt.xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
    plt.ylabel(r'$P\, [\mathrm{Mpc}^3]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

def render_2D_gif(sim_path, out_path):
    render_2D_lst = get_2D_pics(sim_path)

    with Image() as gif:
        for filename in render_2D_lst:
            with Image(filename=filename) as img:
                gif.sequence.append(img)

        for cursor in range(len(gif.sequence)):
            with gif.sequence[cursor] as frame:
                frame.delay = 24
                frame.resize(1000, 1000)
                # Set layer type
                gif.type = 'optimize'
            
            # Converting to django contentfile
            gif.format = 'gif'
            gif.save(filename=out_path)

def render_3D_gif(sim_path, out_path):
    render_3D_lst = get_3D_pics(sim_path)

    with Image() as gif:
        for filename in render_3D_lst:
            with Image(filename=filename) as img:
                gif.sequence.append(img)

        for cursor in range(len(gif.sequence)):
            with gif.sequence[cursor] as frame:
                frame.delay = 24
                frame.resize(1000, 1000)
                # Set layer type
                gif.type = 'optimize'
            
            # Converting to django contentfile
            gif.format = 'gif'
            gif.save(filename=out_path)

def get_2D_pics(sim_path):
    """
    Note for now it is sorted by the time of the file creation so if you copy paste or something
    it will mess up
    """
    # 2D video
    render_2D_lst = []
    for filename in sorted(glob.glob(f'{sim_path}/render2D*'), key=os.path.getmtime):
        render_2D_lst.append(filename)
    
    return render_2D_lst

def get_3D_pics(sim_path):
    """
    Note for now it is sorted by the time of the file creation so if you copy paste or something
    it will mess up
    """
    # 3D video
    render_3D_lst = []
    for filename in sorted(glob.glob(f'{sim_path}/render3D*'), key=os.path.getmtime):
        render_3D_lst.append(filename)

    return render_3D_lst

def plot_sim_pl_2d(sim_snap):
    fig, axs = plt.subplots(2)
    axs[0].scatter(sim_snap.pos_2d_subsample[:, 0], sim_snap.pos_2d_subsample[:, 1], s=3)

    for landscape in sim_snap.pl_2d_lst:
        axs[1].plot(sim_snap.pl_x_2d, landscape)

    return fig, axs

def plot_sim_pl_3d(sim_snap):
    fig = plt.figure()

    # First subplot is 3D scatter
    ax = fig.add_subplot(3, 1, 1, projection='3d')

    ax.scatter(sim_snap.pos_subsample[:, 0], sim_snap.pos_subsample[:, 1], sim_snap.pos_subsample[:, 2], s=1)

    # Second subplot is persistance homology in degree 1
    ax = fig.add_subplot(3, 1, 2)

    for landscape in sim_snap.pl_1_lst_3d:
        ax.plot(sim_snap.pl_x_1_3d, landscape)

    # Third subplot is persistance homology in degree 1
    ax = fig.add_subplot(3, 1, 3)
    for landscape in sim_snap.pl_2_lst_3d:
        ax.plot(sim_snap.pl_x_2_3d, landscape)
    
    return ax

def visual_data_pl(sim_snap):
    """ 
    Parameters
    -----------------
    sim_snap : cwtda.dataio.ConceptSimulationSnap 
        Simulation snap we want the data of 
    
    """
    axs = np.array([
        [plt.subplot(321), plt.subplot(322)],
        [plt.subplot(323), plt.subplot(324, projection='3d')],
        [plt.subplot(325), plt.subplot(326)]
    ])

    pic_2D = PIL.Image.open(sim_snap.pic_2D_filepath)
    axs[0, 0].imshow(pic_2D)
    pic_3D = PIL.Image.open(sim_snap.pic_3D_filepath)
    axs[0, 1].imshow(pic_3D)

    axs[1, 0].scatter(sim_snap.pos_subsample[:, 0], sim_snap.pos_subsample[:, 1], s=1)

    axs[1, 1].scatter(sim_snap.pos_subsample[:, 0], sim_snap.pos_subsample[:, 1], sim_snap.pos_subsample[:, 2], s=1)
    
    axs[2, 1].set_ylim([0, 7])
    for landscape in sim_snap.pl_1_lst_3d:
        axs[2, 0].plot(sim_snap.pl_x_1_3d, landscape)
    
    axs[2, 0].set_ylim([0, 15])
    for landscape in sim_snap.pl_2_lst_3d:
        axs[2, 1].plot(sim_snap.pl_x_2_3d, landscape)

# def gif_a_folder(folder_path):
#     fnames = os.listdir(folder_path)

#     with Image() as gif:
#         for filename in render_3D_lst:
#             with Image(filename=filename) as img:
#                 gif.sequence.append(img)

#         for cursor in range(len(gif.sequence)):
#             with gif.sequence[cursor] as frame:
#                 frame.delay = 24
#                 frame.resize(1000, 1000)
#                 # Set layer type
#                 gif.type = 'optimize'
            
#             # Converting to django contentfile
#             gif.format = 'gif'
#             gif.save(filename=out_path)