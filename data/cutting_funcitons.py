"""
Series of cutting and scaling functions for mimicing broken AT-TPC tracks
with simulated data

Author: Ben Wagner
Date Created: 11 Feb 2025
Date Edited:  13 Feb 2025
"""


import numpy as np


def center_cut(ev, k):
    '''
    Cuts the closest k points to the line (0, 0, z)

    Parameters:
        ev - complete event
        k - number of points to cut from ev

    Returns:
        cut - cut event
    '''
    
    # Get distances
    dists = np.sqrt(np.square(ev[:, 0]) + np.square(ev[:, 1]))
    
    # Sort and cut by distances
    idxs = np.argsort(dists)
    cut = ev[idxs[k:]]

    # Shuffle points in event
    np.random.shuffle(cut)
    return cut


def rand_cut(ev, k, generator):
    '''
    Cuts a random selection of k points that lie consecutively in the z-axis

    Parameters:
        ev - complete event
        k - number of points to cut from ev
        generator - numpy random number generator

    Returns
        cut - cut event
    '''

    # Sort by z
    sorted_ev = ev[ev[:, 2].argsort()]

    # Get random index to cut from and cut
    idx = generator.integers(low=0, high=len(ev)-k)
    cut = np.concatenate((sorted_ev[:idx], sorted_ev[idx+k:]), axis=0)

    # Shuffle points in event
    generator.shuffle(cut, axis=0)
    return cut


def middle_cut(ev, k):
    '''
    Sorts points by z coordinate and removes the middle k points

    Parameters:
        ev - complete events
        k - number of points to cut from ev

    Returns:
        cut - cut event
    '''
    
    # Sort by z
    sorted_ev = ev[ev[:,2].argsort()]

    # Get cut index
    low = (len(ev) - k) // 2
    high = low + k

    # Cut event and shuffle points in event
    cut = sorted_ev[low:high]
    np.random.shuffle(cut)

    return cut


def down_sample(ev, k):
    '''
    Removes a random selection of k points from an event

    Parameters:
        ev - complete event
        k - number of points to cut from ev

    Returns:
        Downsmapled event
    '''

    np.random.shuffle(ev)

    return ev[k:]


class MinMaxScaler:
    '''
    Scales point features to (0, 1) - lns charge amplitude before min/max 
    scaling

    Parameters:
        config - dict containing min and max values of x, y, z, and ln(q)
    '''

    def __init__(self, config):
        self.min_x = config['MIN_X']
        self.min_y = config['MIN_Y']
        self.min_z = config['MIN_Z']
        self.min_lnq = config['MIN_LNQ']

        self.max_x = config['MAX_X']
        self.max_y = config['MAX_Y']
        self.max_z = config['MAX_Z']
        self.max_lnq = config['MAX_LNQ']

        self.x_diff = self.max_x - self.min_x
        self.y_diff = self.max_y - self.min_y
        self.z_diff = self.max_z - self.min_z
        self.lnq_diff = self.max_lnq - self.min_lnq


    def down(self, event):
        '''
        Downscales event

        Parameters:
            event - event to be scaled

        Returns:
            downscaled - downscaled event
        '''
        dxs = (event[:, 0] - self.min_x) / self.x_diff
        dys = (event[:, 1] - self.min_y) / self.y_diff
        dzs = (event[:, 2] - self.min_z) / self.z_diff
        dqs = np.log(event[:, 3])
        dqs = (dqs - self.min_lnq) / self.lnq_diff

        downscaled = np.stack([dxs, dys, dzs, dqs], axis=-1)

        return downscaled


    def up(self, event):
        '''
        Upscales event

        Parameters:
            event - event to be scaled

        Returns:
            upscaled - upscaled event
        '''
        uxs = event[:, 0] * self.x_diff + self.min_x
        uys = event[:, 1] * self.y_diff + self.min_y
        uzs = event[:, 2] * self.z_diff + self.min_z
        uqs = event[:, 3] * self.lnq_diff + self.min_lnq
        uqs = np.exp(uqs)

        upscaled = np.stack([uxs, uys, uzs, uqs], axis=-1)

        return upscaled