"""
Processes and cuts .h5 files of simulated AT-TPC data
into one numpy array for track completion

Author: Ben Wagner
Date Created: 12 Feb 2025
Date Edited:  13 Feb 2025
"""


import numpy as np
import h5py
import sys
from sklearn.model_selection import train_test_split
from cutting_funcitons import center_cut, rand_cut
sys.path.append("/data")


def process_file(file_path):
    """
    Turns .h5 file into one big numpy array. Hard coded for 4 dimensional
    output point cloud and input point cloud to be [x, y, z, t, q_a, ...]

    Parameters:
        file_path: str - path to .h5 file
    
    Returns:
        data: numpy.ndarray - data from .h5 file
        lengths: numpy.ndarray - number of unique points in each event
    """

    file = h5py.File(file_path, 'r')
    keys = list(file.keys())

    lengths = np.ndarray((len(keys)), dtype=int)
    data = np.zeros((len(lengths), np.amax(lengths), 4), dtype=float)

    for idx, k in enumerate(keys):
        
        ev = file[k]
        
        lengths[idx] = len(ev)
        for i, p in enumerate(ev):
            data[idx, i, 0] = p[0]
            data[idx, i, 1] = p[1]
            data[idx, i, 2] = p[2]
            data[idx, i, 3] = p[4]

    return data, lengths


def sample(data, lengths, n_complete, rng):
    """
    Samples events to a set number of total points

    Parameters:
        data: numpy.ndarray - data to sample
        lengths: numpy.ndarray - number of unique points in each event
        n_complete: int - number of points to sample to
        rng: numpy.random._generator.Generator - random number generator 
            (usually an instance of numpy.random.default_rng())

    Returns:
        sampled: numpy.ndarray - sampled events
    """

    sampled = np.ndarray((len(lengths), n_complete, 4))
    ZERO = np.array([0.0, 0.0, 0.0])

    idx = 0
    for i, ev in enumerate(data):

        if lengths[i] == 0: # Discard
            continue

        elif lengths[i] < n_complete: # Up sample

            # Get complete event
            og_l, l = lengths[i], lengths[i]
            sampled[idx, :og_l] = ev[:og_l]

            # Randomly select non zero points to upsample with
            while l < n_complete:
                chosen = rng.choice(ev[:og_l])
                while np.array_equal(chosen, ZERO):
                    chosen = rng.choice(ev[:og_l])
                sampled[idx, l] = chosen
                l += 1

        elif lengths[i] > n_complete: # Down sample

            # Randomly select n_complete points
            whole_ev = ev[:lengths[i]]
            rng.shuffle(whole_ev, axis=0)
            sampled[idx] = whole_ev[:n_complete]

        else: # No sampling necessary
            sampled[idx] = ev[:n_complete]
        
        idx += 1

    return sampled


def filter_invalid(data, lengths, n_complete, min_n_unique):
    """
    Filters events to a minimum number of unique points

    Parameters:
        data: numpy.ndarray - events to be filtered
        lengths: numpy.ndarray - number of unique points per event
        n_complete: int - number of points in a complete event
        min_n_unique: int - mininum number of unique points in an event

    Returns:
        filtered: numpy.ndarray - filtered data
        filtered_lengths: numpy.ndarray - number of unique points in each event
    """

    filtered = np.zeros((np.count_nonzero(np.where(lengths < min_n_unique, 0, lengths)), n_complete, 4))
    filtered_lengths = np.ndarray((len(filtered)), dtype=int)

    idx = 0
    for ev in data:
        n_unique = len(np.unique(ev, axis=0))
        if n_unique < min_n_unique:
            continue
    
        filtered[idx] = ev
        filtered_lengths[idx] = n_unique
        idx += 1

    return filtered, filtered_lengths


def scale_data(data, ranges):
    """
    Min/Max scales data based on ranges (logs qs first)

    Parameters:
        data: numpy.ndarray - data to scale
        ranges: dict - contains min and max values of x,y,z, and ln(q)

    Returns:
        scaled: numpy.ndarray - scaled data
    """

    scaled = np.ndarray(data.shape)

    for i, ev in enumerate(data):
        xs, ys, zs, qs = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]

        dxs = (xs - ranges['MIN_X']) / (ranges['MAX_X'] - ranges['MIN_X'])
        dys = (ys - ranges['MIN_Y']) / (ranges['MAX_Y'] - ranges['MIN_Y'])
        dzs = (zs - ranges['MIN_Z']) / (ranges['MAX_Z'] - ranges['MIN_Z'])
        dqs = (np.log(qs) - ranges['MIN_LNQ']) / (ranges['MAX_LNQ'] - ranges['MIN_LNQ'])

        scaled[i, :, 0] = dxs
        scaled[i, :, 1] = dys
        scaled[i, :, 2] = dzs
        scaled[i, :, 3] = dqs

    return scaled



if __name__ == '__main__':

    SIM_MG_FILE_PATH = "/data/22Mg/point_clouds/simulated/output_digi_HDF_Mg22_Ne20pp_8MeV.h5"
    SIM_O_FILE_PATH  = "/data/16O/point_clouds/simulated/output_digi_HDF_2Body_2T.h5"

    TRAIN_FEATS_SAVE_PATH = "/home/DAVIDSON/bewagner/data/22Mg_16O_combo/simulated/2048c/1536p/Mg_O_combo_train_feats.npy"
    VAL_FEATS_SAVE_PATH = "/home/DAVIDSON/bewagner/data/22Mg_16O_combo/simulated/2048c/1536p/Mg_O_combo_val_feats.npy"
    TEST_FEATS_SAVE_PATH = "/home/DAVIDSON/bewagner/data/22Mg_16O_combo/simulated/2048c/1536p/Mg_O_combo_test_feats.npy"

    TRAIN_LABELS_SAVE_PATH = "/home/DAVIDSON/bewagner/data/22Mg_16O_combo/simulated/2048c/Mg_O_combo_train_labels.npy"
    VAL_LABELS_SAVE_PATH = "/home/DAVIDSON/bewagner/data/22Mg_16O_combo/simulated/2048c/Mg_O_combo_val_labels.npy"
    TEST_LABELS_SAVE_PATH = "/home/DAVIDSON/bewagner/data/22Mg_16O_combo/simulated/2048c/Mg_O_combo_test_labels.npy"

    N_COMPLETE = 2048
    N_PARTIAL = 1024 + 512
    MIN_N_UNIQUE = 128

    RANGES = {
        'MIN_X' : -270.0,
        'MAX_X' :  270.0,
        'MIN_Y' : -270.0,
        'MAX_Y' :  270.0,
        'MIN_Z' : -185.0,
        'MAX_Z' : 1185.0,
        'MIN_LNQ' :  1.0,
        'MAX_LNQ' : 10.2
    }

    RNG = np.random.default_rng()

    # Process
    mg_data, mg_lengths = process_file(SIM_MG_FILE_PATH)
    o_data, o_lengths = process_file(SIM_O_FILE_PATH)

    # Sample
    mg_sampled = sample(mg_data, mg_lengths, N_COMPLETE, RNG)
    o_sampled = sample(o_data, o_lengths, N_COMPLETE, RNG)

    # Filter
    mg_filtered, mg_filt_lens = filter_invalid(mg_sampled, mg_lengths, N_COMPLETE, MIN_N_UNIQUE)
    o_filtered, o_filt_lens = filter_invalid(o_sampled, o_lengths, N_COMPLETE, MIN_N_UNIQUE)

    # Combine datasets
    labels = np.vstack((mg_filtered, o_filtered))

    # Cut Data 3 ways: Center cut, random cut, random sample
    center = np.ndarray((len(labels), N_PARTIAL, 4))
    random_cut = np.ndarray((len(labels), N_PARTIAL, 4))
    random_samp = np.ndarray((len(labels), N_PARTIAL, 4))

    for i, ev in enumerate(labels):
        center[i] = center_cut(ev, (0,0), N_COMPLETE-N_PARTIAL)
        random_cut[i] = rand_cut(ev, N_COMPLETE-N_PARTIAL, RNG)
        sampled = ev
        RNG.shuffle(sampled)
        random_samp[i] = sampled[:N_PARTIAL]

    # Combine and scale
    labels = np.vstack((labels, labels, labels))
    feats = np.vstack((center, random_cut, random_samp))

    labels = scale_data(labels, RANGES)
    feats = scale_data(feats, RANGES)

    # Split into train, val, test sets (60, 20, 20)
    train_feats, val_feats, train_labels, val_labels = train_test_split(feats, labels, test_size=0.2)
    train_feats, test_feats, train_labels, test_labels = train_test_split(train_feats, train_labels, test_size=0.25)

    # Save datasets
    np.save(TRAIN_FEATS_SAVE_PATH, train_feats)
    np.save(VAL_FEATS_SAVE_PATH, val_feats)
    np.save(TEST_FEATS_SAVE_PATH, test_feats)

    np.save(TRAIN_LABELS_SAVE_PATH, train_labels)
    np.save(VAL_LABELS_SAVE_PATH, val_labels)
    np.save(TEST_LABELS_SAVE_PATH, test_labels)

