import os 
import glob
import re
import logging
import sys
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from .. import learn
from .. import util

class ConceptSimulationSnap():
    def __init__(self, snapshot_fn, pwr_fn):
        snap_data = h5py.File(snapshot_fn, 'r')

        # Used to get simulation time and a time
        with open(pwr_fn, 'r') as f:
            ln = f.readline()
            time_gyr = re.search(r't = \d+\.\d+', ln)
            time_gyr = time_gyr.group(0)
            self.time_gyr = float(re.sub(r't = ', '', time_gyr)) # Time in Gyr

            try:
                time_a = re.search(r'a = \d+\.\d+', ln)
                time_a = time_a.group(0)
            except:
                # If errors out then the a is just 1
                time_a = "a = 1"
            self.time_a = float(re.sub(r'a = ', '', time_a))

            job = re.search(r'job \d+', ln).group(0)
            job = re.sub(r'job ', '', job)
            self.job = job

        # Opening logs to find parameter file 
        with open(os.path.join(config.CONCEPT_INSTALL_DIR, 'logs', job)) as f:
            for i in range(30):
                ln = f.readline()

                param_file = re.search(r'copied to "params/\.\d+', ln)
                if param_file != None:
                    param_file = param_file.group(0)
                    param_file = re.sub(r'copied to "params/', '', param_file)
                    self.sim_name = param_file
                    param_filepath = os.path.join(config.CONCEPT_INSTALL_DIR, 'params', param_file)
                
                    param_class = re.search(r'Parameter file: "params/([^\s]+)', ln)
                    param_class = param_class.group(0)
                    param_class = re.sub(r'Parameter file: "params/', '', param_class)
                    param_class = re.sub(r'"', '', param_class)
                    self.param_class = param_class
                    break

            if not isinstance(param_file, str):
                raise Exception(f"Could not find parameter file for simulation {job}")

        self.concept_sim_dir = os.path.join(config.CONCEPT_INSTALL_DIR, "output", self.sim_name)
        self.pic_2D_filepath = os.path.join(self.concept_sim_dir, f"render2D_a={self.time_a}.png")
        self.pic_3D_filepath = os.path.join(self.concept_sim_dir, f"render3D_a={self.time_a}.png")

        for group in snap_data.keys():
            for dset in snap_data[group].keys() :
                ds_data = snap_data[group][dset] # returns HDF5 dataset object
                pos = np.c_[ds_data['posx'], ds_data['posy'], ds_data['posz']]
        
        self.pos = pos 
        self.pos_subsample = None

        # Iterating through parameter file and getting log data
        with open(param_filepath) as f:
            fread = f.read()
            size = re.search(r'^_size = \d+', fread, re.MULTILINE)
            if size != None:
                size = size.group(0)
                self.size = float(re.sub(r'_size = ', '', size))

            boxsize = re.search(r'^boxsize = \d+\*Mpc', fread, re.MULTILINE)
            if boxsize != None:
                boxsize = boxsize.group(0)
                self.boxsize = re.sub(r'boxsize = ', '', boxsize)

            self.hubble_constant = None
            hubble_str = re.search(r'^H0 += \d+\*km/\(s\*Mpc\)', fread, re.MULTILINE)
            if hubble_str != None:
                hubble_str = hubble_str.group(0)
                self.hubble_str = re.sub(r'H0 += ', '', hubble_str)
                self.hubble_constant = float(re.sub(r'\*km/\(s\*Mpc\)', '', self.hubble_str))
        
            self.omega_b = None
            omega_b_str = re.search(r'^Ωb += \d+.\d+', fread, re.MULTILINE)
            if omega_b_str != None:
                omega_b_str = omega_b_str.group(0)
                self.omega_b = float(re.sub(r'Ωb += ', '', omega_b_str))

            self.omegacdm = None
            omegacdm_str = re.search(r'^Ωcdm += \d+.\d+', fread, re.MULTILINE)
            if omegacdm_str != None:
                omegacdm_str = omegacdm_str.group(0)
                self.omegacdm = float(re.sub(r'Ωcdm += ', '', omegacdm_str))
            
            self.a_begin = None
            a_begin_str = re.search(r'^Ωcdm += \d+.\d+', fread, re.MULTILINE)
            if a_begin_str != None:
                a_begin_str = a_begin_str.group(0)
                self.a_begin = float(re.sub(r'Ωcdm += ', '', a_begin_str))

    def subsample(self, subsample_percent=5e-4, idxs=None):
        if idxs is None:
            n_pts = int(self.pos.shape[0] * subsample_percent)
            idxs = np.floor(np.random.choice(range(self.pos.shape[0] - 1), n_pts, replace=False)).astype(int)

        self.subsample_idxs = idxs
        self.pos_subsample = self.pos[idxs]

    def create_pl(self, max_pts=1000, sample_range=[10, 50], resolution=100, mode='cube_knn'):
        """ 
        Parameters
        -------------------
        max_pts : int
            Maximum number of points allowable for the computation. This should be specified to avoid 
            long load times

        sample_range ([float, float]) – minimum and maximum of all piecewise-linear function domains, of the form [x_min, x_max] 
        (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, 
        it can be computed from the persistence diagrams with the fit() method.
        
        resolution : int
            Number of sample for all piecewise-linear functions (default 100).

        mode : str
            options are standard, subsample, cube and cube_knn. Standard will compute the persistance landscape of the entire pos array. This 
            is usually not recommended and is pretty much impossible for actual hydrodyanmic simulations. Subsample will run it on 
            the the subsampled points from subsample. Cube will run it on a cubical section that that was created with cut_cube.

        
        Saves
        -------------------
        self.pl_x_1 : np.array
            samples on interval for parameters in persistance landscape for Homology in degree 1
        
        self.pl_x_2 : np.array
            samples on interval for parameters in persistance landscape for Homology in degree 2

        self.pl_1_lst : list of np.array
            Piecewise linear persistance landscapes in degree 1
        
        self.pl_2_lst : list of np.array
            Piecewise linear persistance landscapes in degree 2
        """

        if mode == "standard":
            if self.pos.shape[0] > max_pts:
                logging.exception("The number of points exceeds the max number of points specified. If you would like to increase this number set max_pts= in the function call. Be careful though, setting this number too high without subsampling will either crash or extremely slow down your system. Call ConceptSimulationSnap.subsample() before running this.")
                sys.exit(0)
            pt_cloud = self.pos

        elif mode == "subsample":
            if self.pos_subsample is None:
                logging.exception("Points have not been subsampled")
                sys.exit(0)
            if self.pos_subsample.shape[0] > max_pts:
                logging.exception("The number of points exceeds the max number of points specified. If you would like to increase this number set max_pts= in the function call. Be careful though, setting this number too high without subsampling will either crash or extremely slow down your system. Call ConceptSimulationSnap.subsample() before running this.")
                sys.exit(0)
            pt_cloud = self.pos_subsample

        elif mode == "cube":
            if self.cube_pos is None:
                logging.exception("Points have not been cubed")
                sys.exit(0)
            if self.cube_pos.shape[0] > max_pts:
                logging.exception("The number of points exceeds the max number of points specified. If you would like to increase this number set max_pts= in the function call. Be careful though, setting this number too high without subsampling will either crash or extremely slow down your system. Call ConceptSimulationSnap.subsample() before running this.")
                sys.exit(0)            
            pt_cloud = self.cube_pos

        elif mode == "cube_knn":
            if self.cube_pos_knn.shape[0] > max_pts:
                logging.exception("The number of points exceeds the max number of points specified. If you would like to increase this number set max_pts= in the function call. Be careful though, setting this number too high without subsampling will either crash or extremely slow down your system. Call ConceptSimulationSnap.subsample() before running this.")
                sys.exit(0)    
            pt_cloud = self.cube_pos_knn

        else:
            logging.exception("Unkown mode, options are standard, subsample and cube")
            sys.exit(0)

        pl_x_1, pl_x_2, pl_1_lst, pl_2_lst = learn.tda.point_cloud_to_pl(pt_cloud, sample_range=sample_range, resolution=resolution)

        # Persistance landscapes in 3 dimensions
        self.pl_x_1, self.pl_x_2, self.pl_1_lst, self.pl_2_lst = pl_x_1, pl_x_2, pl_1_lst, pl_2_lst
    
    def cut_cube(self, lower_bound=50, upper_bound=90):
        lb, ub = lower_bound, upper_bound
        self.cube_pos = np.array([pt for pt in self.pos if (lb < pt[0] < ub and lb < pt[1] < ub and lb < pt[2] < ub)])

    def knn_denoise(self, mode="cube", n_neighbors=10):
        """ Denoises cube position using k-Nearest-Neighbors """
        if mode == "cube":
            self.cube_pos_knn = learn.tda.knn_denoise(self.cube_pos, n_neighbors)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def save(self, filepath="fill"):

        if not os.path.exists(os.path.join(config.CWTDA_DIR, "data", "concept", self.sim_name)):
            os.makedirs(os.path.join(config.CWTDA_DIR, "data", "concept", self.sim_name))

        if filepath == "fill":
            filepaths = os.listdir(os.path.join(config.CWTDA_DIR, "data", "concept", self.sim_name))
            int_fpath = []
            for fpath in filepaths:

                fpath = re.search(r"\d+.pkl", fpath)
                if fpath == None:
                    continue
                fpath = fpath.group(0)
                fpath = re.sub(r".pkl", '', fpath)
                int_fpath.append(int(fpath))

            if int_fpath == []:
                int_fpath.append(0)
            new_name = max(int_fpath) + 1

            filepath = os.path.join(config.CWTDA_DIR, "data", "concept", self.sim_name, str(new_name)+".pkl")
            self.filepath = filepath
        
        elif filepath == "base":
            if self.filepath == None:
                raise Exception('File has not been saved before, use filepath="fill" to create a new name')
        
        else:
            self.filepath == filepath
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self, f)
            logging.info(f"ConceptSimulationSnap object saved to filepath {self.filepath}")
        except:
            logging.warning(f"ConceptSimulationSnap object was not able to be saved to filepath {self.filepath}")


def load_data(path, mode):
    """ 
    Parameters
    -------------------
    path : str or POSIXPATH
        Path to simulation or ConceptSimulationSnap pickled objects

    mode : str
        Either BaseSimulation, AnalyzedSimulation, AllAnalyzed. If BaseSimulation is given, the path should the direct output of a 
        concept simulation. If AnalyzedSimulation is given, the path should be in cwtda/data/concept/INSERTFNAMEHERE. If AllAnalyzed
        is given, the path should be where the concept simulation is and the simulation folders are stored
    
    Returns
    -------------------
    concept_sim_snap_lst : list of ConceptSimulationSnap objects
        It will either be already analyzed or not analyzed

    pl_2_lst : list of np.array

    """

    if mode == 'BaseSimulation':
        snapshot_fn_lst = [f for f in sorted(glob.glob(f'{path}/snapshot*'), key=os.path.getmtime)]
        pwr_fn_lst = [f for f in sorted(glob.glob(f'{path}/powerspec*'), key=os.path.getmtime)]

        concept_sim_snap_lst = []
        for snapshot_fn, pwr_fn in zip(snapshot_fn_lst, pwr_fn_lst):
            concept_sim_snap_lst.append(ConceptSimulationSnap(snapshot_fn, pwr_fn))

    elif mode == 'AnalyzedSimulation':
        concept_sim_snap_lst = []
        for fname in os.listdir(path):
            loadpath = os.path.join(path, fname)
            concept_sim_snap_lst.append(ConceptSimulationSnap.load(loadpath))
    
    elif mode == 'AllAnalyzed':
        concept_sim_snap_lst = []
        for folder_name in os.listdir(path):
            for loadname in os.listdir(os.path.join(path, folder_name)):
                loadpath = os.path.join(path, folder_name, loadname)
                concept_sim_snap_lst.append(ConceptSimulationSnap.load(loadpath))

    else:
        raise Exception("Unrecognized mode. Options are BaseSimulation, AnalyzedSimulation or AllAnalyzed")
    
    return concept_sim_snap_lst
