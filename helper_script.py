import matplotlib.pyplot as plt

from cwtda.tda import kinematic_voronoi_foam_analysis as kvfa 
from cwtda.kinematic_voronoi_foams import kinematic_voronoi as kv 

data_folder_name = 'test'

# for i in range(10, 100):
#     data_folder_name = str(i)
#     kv.generate_time_series_voronoi_2d(mean_density=100, num_nuclei=8, simulation_name=data_folder_name)
#     kvfa.get_persistance_landscapes(data_folder_name)

# kvfa.plot_persistance('0')
# pl_x, pl = kvfa.load_persistance_landscapes()
# pl_x = pl_x[0]
# mean_15 = kvfa.get_average_persistance_landscape(15)
# mean_1 = kvfa.get_average_persistance_landscape(1)

# plt, axs = plt.subplots(2, figsize=(8,6), sharex=True, sharey=True)
# axs[0].set_title("Mean Landscape Epoch=0")
# axs[1].set_title("Mean Landscape Epoch=15")
# kvfa.plot_persistance_landscape(pl_x, mean_1, ax=axs[0])
# kvfa.plot_persistance_landscape(pl_x, mean_15, ax=axs[1], show=True)

pl_x_lst, pl_lst = kvfa.load_persistance_landscapes()
kvfa.nn(pl_lst, show_score=True)

 #ffmpeg -r 5 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4