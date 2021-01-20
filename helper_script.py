import os 

import matplotlib.pyplot as plt

from cwtda.learn import kvfa 
from cwtda.learn import tda
from cwtda.concept import dataio
from cwtda import config

# kvfa.plot_persistance('1')

concept_sim_snap_lst = dataio.load_data("/home/n/Documents/Research/etc_codes/CONCEPT/concept/output/test_pl")
sim_snap = concept_sim_snap_lst[0]
sim_snap.create_pl()

#ffmpeg -r 5 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4