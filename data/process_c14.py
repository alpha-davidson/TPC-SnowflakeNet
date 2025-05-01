import numpy as np
import h5py
import sys
import random
import os
sys.path.append("/data")
sys.path.append("../")


def process_file(file_path, save_path, min_len, max_len):
    '''
    Turns .h5 file into individual numpy arrays for each event. 
    Hard coded for 4 dimensional output point cloud and 
    input point cloud to be [x, y, z, t, q_a, ...]

    Parameters:
        file_path: str - Path to .h5 file
        save_path: str - Path to folder for saving .npy files for each event
        min_len: int - Minimum number of unique points allowed in an event
        max_len: int - Maximum number of unique points allowed in an event

    Returns:
        None
    '''

    outer = h5py.File(file_path, 'r')
    file = outer['cloud']
    keys = list(file.keys())

    lengths = np.ndarray((len(keys)), dtype=int)
    for i, k in enumerate(keys):
        lengths[i] = len(file[k])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    count = 0
    for i, k in enumerate(file):

        if lengths[i] < min_len or lengths[i] > max_len:
            continue

        event = np.ndarray((lengths[i], 4))
        for idx, p in enumerate(file[k]):
            event[idx, 0] = p[0]
            event[idx, 1] = p[1]
            event[idx, 2] = p[2]
            event[idx, 3] = p[3]

        if np.any(np.isnan(event)) or np.any(np.isinf(event)):
            count += 1
            print(f"NaN or Inf in {count} events")
            continue
        # Save each event with a random hash
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, event)
        

def make_category_file(data, path, experiment):

    with open(path, 'w') as jason:

        jason.write("[\n")
        jason.write("\t{\n")
        jason.write("\t\t\"experiment\": \""+experiment+"\",\n")
        jason.write("\t\t\"data\": [\n")

        for i, event in enumerate(data):
            if i == (len(data) - 1):
                jason.write(f"\t\t\t\"{event.split('.')[0]}\"]\n")
                break
            jason.write(f"\t\t\t\"{event.split('.')[0]}\",\n")

        jason.write("\t}\n]")

    return


if __name__ == "__main__":


    C_FILE_PATH = '/data/14C/pointcloud_bazin/run_0055.h5'

    C_SAVE_PATH = '/home/DAVIDSON/bewagner/TPC-SnowflakeNet/data/c14/'


    MIN_N_POINTS = 50
    MAX_N_POINTS = 750

    CATEGORY_FILE_PATH = 'completion/category_files/c14.json'


    process_file(C_FILE_PATH, C_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS)
    make_category_file(os.listdir(C_SAVE_PATH), CATEGORY_FILE_PATH, "14C")