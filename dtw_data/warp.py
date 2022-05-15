from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np

s1 = np.array([0.340435, 0.436146, 0.56870, 0.532383, 0.537291, 0.333444])
s2 = np.array([0.31219, 0.43387, 0.223076, 0.461766, 0.350872, 0.41279])
# s2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# print(dtw.distance(s1, s2))
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="warp.svg")
