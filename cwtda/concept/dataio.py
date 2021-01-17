import os 
import glob
import re

import h5py
import numpy as np

from .. import config

class ConceptSimulationSnap():
    def __init__(self, snapshot_fn, pwr_fn):
        snap_data = h5py.File(snapshot_fn, 'r')

        # Used to get simulation time and a time
        with open(pwr_fn, 'r') as f:
            ln = f.readline()
            time_gyr = re.search(r't = \d+\.\d+', ln)
            time_gyr = time_gyr.group(0)
            self.time_gyr = float(re.sub(r't = ', '', time_gyr)) # Time in Gyr

            time_a = re.search(r'a = \d+\.\d+', ln)
            time_a = time_a.group(0)
            self.time_a = float(re.sub(r'a = ', '', time_a))

            job = re.search(r'job \d+', ln).group(0)
            job = re.sub(r'job ', '', job)

        # Opening logs to find parameter file 
        with open(os.path.join(config.CONCEPT_INSTALL_DIR, 'logs', job)) as f:
            for i in range(30):
                ln = f.readline()

                param_file = re.search(r'copied to "params/\.\d+', ln)
                if param_file != None:
                    param_file = param_file.group(0)
                    param_file = re.sub(r'copied to "params/', '', param_file)
                    param_filepath = os.path.join(config.CONCEPT_INSTALL_DIR, 'params', param_file)
                    break 
            
            if not isinstance(param_file, str):
                raise Exception(f"Could not find parameter file for simulation {path}")

        for group in snap_data.keys():
            for dset in snap_data[group].keys() :
                ds_data = snap_data[group][dset] # returns HDF5 dataset object
                pos = np.c_[ds_data['posx'], ds_data['posy'], ds_data['posz']]
        
        self.pos = pos 

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

def load_data(path):
    snapshot_fn_lst = [f for f in sorted(glob.glob(f'{path}/snapshot*'), key=os.path.getmtime)]
    pwr_fn_lst = [f for f in sorted(glob.glob(f'{path}/powerspec*'), key=os.path.getmtime)]

    for snapshot_fn, pwr_fn in zip(snapshot_fn_lst, pwr_fn_lst):
        ConceptSimulationSnap(snapshot_fn, pwr_fn)