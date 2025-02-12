import numpy as np
import os
import sys
import cutting_funcitons as cf
import json
import tqdm


def get_data(category_file_path):

    with open(category_file_path, 'r') as j:
        data = json.load(j)

    mg_data, o_data = data[0], data[1]

    return mg_data, o_data


def sample_complete_point_clouds(data_dir_path, npoints, data):

    for h in tqdm.tqdm(os.listdir(data_dir_path)):

        cloud = np.load(data_dir_path+'/'+h)
        length = len(cloud)

        sampled = np.ndarray((npoints, 4))

        if length < npoints:
            # UpSample
            pass
        elif length > npoints:
            # DownSample
            pass
        else:
            # NoSampling
            pass

        hsh = h.split('.')[0]
        if hsh in data[0]['train'] or hsh in data[1]['train']:
            np.save(f"./variable_length/train/complete")