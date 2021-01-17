import glob, os, re
import numpy as np
import matplotlib.pyplot as plt

from wand.image import Image

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
    print(render_3D_lst)

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