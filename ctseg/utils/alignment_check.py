import numpy as np


def eval_align(label, element):
    x, y, z = np.where(label == element)
    x_max, y_max, z_max = label.shape
    if len(z) == 0:
        print("No nonzero elements")
        return 0
    
    
    print("prepping data...")
    tgt_x = np.ones((x_max,y_max,z.max() + 1)) * np.nan
    tgt_y = np.ones((x_max,y_max,z.max() + 1)) * np.nan

    for i in range(len(x)):
        tgt_x[x[i], y[i], z[i]] = x[i]
        tgt_y[x[i], y[i], z[i]] = y[i]
        
    print("calculating COM...")
        
    x_mean = np.nanmean(tgt_x, axis=(0,1))
    x_mean = x_mean[~np.isnan(x_mean)]
    y_mean = np.nanmean(tgt_y, axis=(0,1))
    y_mean = y_mean[~np.isnan(y_mean)]
    
    print("calculating median of perimeter...")
    l_x_max = np.nanmean(np.nanmax(tgt_x, axis=(0)), axis=0)
    l_x_max = l_x_max[~np.isnan(l_x_max)]
    l_x_min = np.nanmean(np.nanmin(tgt_x, axis=(0)), axis=0)
    l_x_min = l_x_min[~np.isnan(l_x_min)]
    l_y_max = np.nanmean(np.nanmax(tgt_y, axis=(1)), axis=0)
    l_y_max = l_y_max[~np.isnan(l_y_max)]
    l_y_min = np.nanmean(np.nanmin(tgt_y, axis=(1)), axis=0)
    l_y_min = l_y_min[~np.isnan(l_y_min)]

    print("calculating diff")
    x_diff = (l_x_max + l_x_min) / 2 - x_mean
    x_diff = x_diff[~np.isnan(x_diff)]
    y_diff = (l_y_max + l_y_min) / 2 - y_mean
    y_diff = y_diff[~np.isnan(y_diff)]
    
    return x_mean, y_mean, x_diff, y_diff
