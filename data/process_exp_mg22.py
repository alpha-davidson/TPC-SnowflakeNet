import numpy as np
import h5py
import random
import sys
import os
sys.path.append("/data")
sys.path.append("../")

def filter_charge(ev, threshold):
    return ev[ev[:, -1] >= threshold]


def is_null(ev, min_n_points=100):

    wedge_pt_count = 0
    for p in ev:
        x = p[0]
        y = p[1]
        
        if (-130 < x < 10 and -10 < y < 130 and (-x/3) < y < (-3*x)):
            wedge_pt_count += 1

    if (len(ev) <= min_n_points) or (wedge_pt_count >= 80) or (wedge_pt_count / len(ev) >= 0.3):
        return True

    return False


def process_file(file_path, save_path, min_len, max_len, min_a=50):

    file = h5py.File(file_path, 'r')
    keys = list(file.keys())

    lengths = np.zeros((len(keys)), dtype=int)
    for i, k in enumerate(keys):
        lengths[i] = len(file[k])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, k in enumerate(keys):

        if lengths[i] < min_len or lengths[i] > max_len:
            continue

        event = np.zeros((lengths[i], 4), dtype=float)
        for idx, p in enumerate(file[k]):
            event[idx, 0] = p[0]
            event[idx, 1] = p[1]
            event[idx, 2] = p[2]
            event[idx, 3] = p[4]

        charge_filt = filter_charge(event, min_a)

        if is_null(charge_filt, min_n_points=min_len):
            continue

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


def main():

    COMPELTE_FILE_PATH = '/data/22Mg/point_clouds/experimental/Mg22_exp.h5'
    SAVE_PATH = 'data/22Mg/'

    CATEGORY_FILE_PATH = 'completion/category_files/mg22_exp.json'

    MAX_N_POINTS = 750
    MIN_N_POINTS = 100 # Typically 50, but autofloor is 100 b/c of is_null()

    MIN_A = 50

    process_file(COMPELTE_FILE_PATH, SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS, min_a=MIN_A)
    make_category_file(os.listdir(SAVE_PATH), CATEGORY_FILE_PATH, '22Mg')
    


if __name__ == '__main__':
    main()