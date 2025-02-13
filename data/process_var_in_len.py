"""
Processes and cuts .h5 files of simulated AT-TPC data
into one numpy array per event for track completion

Author: Ben Wagner
Date Created: 2/12/25

"""


import numpy as np
import json
import os
import h5py
import random
import sys
import cutting_funcitons as cf
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

    file = h5py.File(file_path, 'r')
    keys = list(file.keys())

    lengths = np.ndarray((len(keys)), dtype=int)
    for i, k in enumerate(keys):
        lengths[i] = len(file[k])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i, k in enumerate(file):

        if lengths[i] < min_len or lengths[i] > max_len:
            continue

        event = np.ndarray((lengths[i], 4))
        for idx, p in enumerate(file[k]):
            event[idx, 0] = p[0]
            event[idx, 1] = p[1]
            event[idx, 2] = p[2]
            event[idx, 3] = p[4]

        # Save each event with a random hash
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, event)


def get_ttv_split(mg_path, o_path, train_split=0.6, val_split=0.2):
    '''
    Splits events into train, val, and test sets

    Parameters:
        mg_path: str - path to 22Mg events
        o_path: str - path to 16O events
        train_split: float - percentage of events to put in train set
        val_split: float - percentage of events to put in val set

    Returns:
        train: np.ndarray - hahses that are part of the train set 
                            (and which isotope they are)
        val: np.ndarray - hashes that are part of the val set
                          (and which isotope they are)
        test: np.ndarray - hashes that are part of the test set
                           (and which isotope they are)
    '''
    
    mg_hashes = os.listdir(mg_path)
    o_hashes = os.listdir(o_path)

    name_arr = np.ndarray((len(mg_hashes) + len(o_hashes)),
                          dtype=[('hash', 'object'), ('experiment', 'object')])
    i = 0
    for h in mg_hashes:
        name_arr[i] = (h.split('.')[0], '22Mg')
        i += 1
    for h in o_hashes:
        name_arr[i] = (h.split('.')[0], '16O')
        i += 1

    rng = np.random.default_rng()
    rng.shuffle(name_arr)

    num_train = int(train_split * i)
    num_val = int(val_split * i)

    train = name_arr[:num_train]
    val = name_arr[num_train:num_train+num_val]
    test = name_arr[num_train+num_val:]

    return train, val, test


def make_category_file(train, val, test, path):
    '''
    Creates .json category file for dataset.
    Needs Tuning -- Commas are missing/misplaced

    Parameters:
        train: np.ndarray - hahses that are part of the train set 
                            (and which isotope they are)
        val: np.ndarray - hashes that are part of the val set
                          (and which isotope they are)
        test: np.ndarray - hashes that are part of the test set
                           (and which isotope they are)
        path: str - path to category file

    Returns:
        None
    '''

    with open(path, 'w') as jason:

        jason.write("[\n")

        jason.write("\t{\n")
        
        jason.write("\t\t\"experiment\": \"22Mg\",\n")
        jason.write("\t\t\"train\": [\n")

        for event in train[:-1]:
            if event['exp'] != "22Mg":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        if train[-1]['exp'] == "22Mg":
            jason.write(f"\t\t\t\"{train[-1]['hash']}\"\n")
        jason.write("\t\t],\n")

        jason.write("\t\t\"val\": [\n")
        for event in val[:-1]:
            if event['exp'] != "22Mg":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        if val[-1]['exp'] == "22Mg":
            jason.write(f"\t\t\t\"{val[-1]['hash']}\"\n")
        jason.write("\t\t],\n")

        jason.write("\t\t\"test\": [\n")
        for event in test[:-1]:
            if event['exp'] != "22Mg":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        if test[-1]['exp'] == "22Mg":
            jason.write(f"\t\t\t\"{test[-1]['hash']}\"\n")
        jason.write("\t\t]\n")

        jason.write("\t},\n")

        jason.write("\t{\n")
        jason.write("\t\t\"experiment\": \"16O\",\n")
        jason.write("\t\t\"train\": [\n")

        for event in train[:-1]:
            if event['exp'] != "16O":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        if train[-1]['exp'] == '16O':
            jason.write(f"\t\t\t\"{train[-1]['hash']}\"\n")
        jason.write("\t\t],\n")

        jason.write("\t\t\"val\": [\n")
        for event in val[:-1]:
            if event['exp'] != "16O":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        if val[-1]['exp'] == '16O':
            jason.write(f"\t\t\t\"{val[-1]['hash']}\"\n")
        jason.write("\t\t],\n")

        jason.write("\t\t\"test\": [\n")
        for event in test[:-1]:
            if event['exp'] != "16O":
                continue
            jason.write(f"\t\t\t\"{event['hash']}\",\n")
        if test[-1]['exp'] == '16O':
            jason.write(f"\t\t\t\"{test[-1]['hash']}\"\n")
        jason.write("\t\t]\n")

        jason.write("\t}\n")

        jason.write("]")
        return
    

def sort_files(mg_path, o_path, train, val, test):
    '''
    Reorganizes files based on their dataset

    Parameters:
        mg_path: str - 
        o_path: str - 
        train: np.ndarray - 
        val: np.ndarray - 
        test: np.ndarray - 

    Returns:
        None
    '''

    for file in os.listdir(mg_path):

        hsh = file.split(".")[0]

        if hsh in train['hash']:
            os.rename(mg_path+file, mg_path+'train/complete/'+file)
        elif hsh in val['hash']:
            os.rename(mg_path+file, mg_path+'val/complete/'+file)
        elif hsh in test['hash']:
            os.rename(mg_path+file, mg_path+'test/complete/'+file)
        else:
            raise NameError()

    for file in os.listdir(o_path):
    
        hsh = file.split(".")[0]
        if hsh in train['hash']:
            os.rename(o_path+file, o_path+'train/complete/'+file)
        elif hsh in val['hash']:
            os.rename(o_path+file, o_path+'val/complete/'+file)
        elif hsh in test['hash']:
            os.rename(o_path+file, o_path+'test/complete/'+file)
        else:
            raise NameError()

    return


def create_partial_clouds(mg_path, o_path, percentage_cut=0.25):
    '''
    Cuts complete cloud into 3 partial clouds: center, random, and downsampled

    Parameters:
        mg_path: str - 
        o_path: str - 
        percentage_cut: float - 

    Returns:
        None
    '''

    rng = np.random.default_rng()

    for split in ('/train', '/val', '/test'):
        for file in os.listdir(mg_path+split+'/complete'):
            event = np.load(file)
            k = len(event) * percentage_cut

            center = cf.center_cut(event, k)
            rand = cf.rand_cut(event, rng)
            down = cf.down_sample(event, k)

            hsh = file.split('.')[0]
            if not os.path.exists(mg_path+split+f'/partial/{hsh}'):
                os.mkdir(mg_path+split+f'/partial/{hsh}')

            np.save(mg_path+split+f'/partial/{hsh}/center.npy', center)
            np.save(mg_path+split+f'/partial/{hsh}/rand.npy', rand)
            np.save(mg_path+split+f'/partial/{hsh}/down.npy', down)

    for split in ('/train', '/val', '/test'):
        for file in os.listdir(o_path+split+'/complete'):
            event = np.load(file)
            k = len(event) * percentage_cut

            center = cf.center_cut(event, k)
            rand = cf.rand_cut(event, rng)
            down = cf.down_sample(event, k)

            hsh = file.split('.')[0]
            if not os.path.exists(o_path+split+f'/partial/{hsh}'):
                os.mkdir(o_path+split+f'/partial/{hsh}')

            np.save(o_path+split+f'/partial/{hsh}/center.npy', center)
            np.save(o_path+split+f'/partial/{hsh}/rand.npy', rand)
            np.save(o_path+split+f'/partial/{hsh}/down.npy', down)

    return


if __name__ == '__main__':

    MG_FILE_PATH = '/data/'
    O_FILE_PATH = '/data/'

    MG_SAVE_PATH = ''
    O_SAVE_PATH = ''

    MIN_N_POINTS = 0
    MAX_N_POINTS = 0

    CATEGORY_FILE_PATH = '../completion/category_files/'

    # Process
    process_file(MG_FILE_PATH, MG_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS)
    process_file(O_FILE_PATH, O_SAVE_PATH, MIN_N_POINTS, MAX_N_POINTS)

    # Split and sort
    train, val, test = get_ttv_split(MG_SAVE_PATH, O_SAVE_PATH)
    make_category_file(train, val, test, CATEGORY_FILE_PATH)
    sort_files(MG_SAVE_PATH, O_SAVE_PATH, train, val, test)

    # Cut
    create_partial_clouds(MG_SAVE_PATH, O_SAVE_PATH)