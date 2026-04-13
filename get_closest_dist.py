import numpy as np

def get_closest_dist(x0, obs_list, rob_rad):
    if len(obs_list) == 0:
        return np.inf, None, np.array([])

    rob_x = x0[0]
    rob_y = x0[1]

    obs_x = obs_list[:, 0]
    obs_y = obs_list[:, 1]
    obs_r = obs_list[:, 2]

    dist_center = np.sqrt((rob_x - obs_x)**2 + (rob_y - obs_y)**2)
    all_dists = dist_center - rob_rad - obs_r

    min_idx = np.argmin(all_dists)
    min_dist = all_dists[min_idx]

    return min_dist, min_idx, all_dists