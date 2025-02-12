import numpy as np
import json
import os
import h5py
import random
import sys
from sklearn.model_selection import train_test_split
import cutting_funcitons as cf


def get_file_and_lengths(file_path):

    file = h5py.File(file_path, 'r')
    keys = list(file.keys())

    lengths = np.ndarray((len(keys)), dtype=int)
    for i, k in enumerate(keys):
        lengths[i] = len(file[k])

    return file, lengths


def get_complete_event(file, lengths, min_len, max_len, save_path):

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
        name = f"{save_path}/{random.getrandbits(128):032x}.npy"
        np.save(name, event)


def get_ttv_split(mg_path, o_path, train_split=0.6, val_split=0.2):
    
    mg_hashes = os.listdir(mg_path)
    o_hashes = os.listdir(o_path)

    name_arr = np.ndarray((len(mg_hashes) + len(o_hashes)), dtype=[('hash', 'object'), ('experiment', 'object')])
    i = 0
    for h in mg_hashes:
        name_arr[i] = (h.split('.')[0], '22Mg')
        i += 1
    for h in mg_hashes:
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
    Needs Tuning -- Fucks up with commas a bit
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