from cwtda.tda import kinematic_voronoi_foam_analysis as kvfa 
from cwtda.kinematic_voronoi_foams import kinematic_voronoi as kv 

data_folder_name = 'test'
# kv.generate_time_series_voronoi_2d(mean_density=100, num_nuclei=8)
kvfa.plot_persistance(data_folder_name)

 #ffmpeg -r 5 -f image2 -s 1920x1080 -i %03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4