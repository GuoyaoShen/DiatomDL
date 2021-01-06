import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


def generate_data(PATH_DATA, NUM_SETSIZE, NUM_PARAM):
    '''
    Generate design params - spectra data.
    :param PATH_DATA: path for txt file, don't include file number
    :param NUM_SETSIZE: number of files in the directory
    :param NUM_PARAM: number of raw design params
    :return: data_param, data_spectra: design params and spectras, in numpy array
    '''


    data_param = np.array([]).reshape(0, NUM_PARAM)
    data_spectra = np.array([]).reshape(0, 1001, 2)

    for idx_file in range(NUM_SETSIZE):
        if idx_file == 19:
            continue

        path_file = PATH_DATA + str(idx_file + 1) + '.txt'

        # read file
        print('========================================= FILE ' + str(
            idx_file + 1) + ' =========================================')
        num_combination = 0
        with open(path_file) as f:
            lines = f.readlines()

            spectra_all = np.array([]).reshape(0, 1001, 2)  # shape for each spectra: [1001,2]
            spectra = np.array([]).reshape(0, 2)
            param_all = np.array([]).reshape(0, NUM_PARAM)

            for i, line in enumerate(lines):
                if (i % 1004 != 0) & (i % 1004 != 1) & (i % 1004 != 2):  # read spectra data
                    line_array = np.fromstring(line, dtype=float, sep=' ')
                    spectra = np.vstack((spectra, line_array))

                if i % 1004 == 0:  # every (3+1001) lines, read param title
                    # param = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", line)]  # extract the float param
                    param = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+(?=;|})", line)]  # extract the float param
                    param = np.array(param)
                    # print('raw', param)
                    # idx_pick = [1,3,5,7,9,11,12,13,14,16,18,20,22,24,26,28,29,30]  # only pick the chosen params
                    # param = param[idx_pick]
                    print(num_combination, param)
                    print('------------')
                    param_all = np.vstack((param_all, param))
                    num_combination += 1

                if i % 1004 == 1003:  # every end of the combination, concat
                    # print(num_combination)
                    spectra_all = np.concatenate((spectra_all, spectra[np.newaxis, ...]), axis=0)
                    spectra = np.array([]).reshape(0, 2)

        # concat data
        data_param = np.concatenate((data_param, param_all), axis=0)
        data_spectra = np.concatenate((data_spectra, spectra_all), axis=0)

    return data_param, data_spectra


def generate_dataset(PATH_ZIPSET, idx_pick_param=[], BTSZ=10):
    '''
    Generate torch dataset and dataloader from zipped numpy dataset.
    :param PATH_ZIPSET: path for zipped numpy dataset
    :param idx_pick_param: list of idx of selected design params, default as empty list
    :param BTSZ: batch size, default as 10
    :return: dataset, dataloader: torch dataset and dataloader
    '''

    data = np.load(PATH_ZIPSET)
    param = data['param']
    spectra_R = data['R'][..., 1]  #[N,1001]
    spectra_T = data['T'][..., 1]
    if idx_pick_param:  # select param
        param = param[..., idx_pick_param]

    # concat reflection and transmission spectras as one
    spectra_R = np.expand_dims(spectra_R, 1)  #[N,1,1001]
    spectra_T = np.expand_dims(spectra_T, 1)
    spectra_RT = np.concatenate((spectra_R, spectra_T), axis=1)  #[N,2,1001]

    tensor_x = torch.Tensor(param)  # transform to torch tensor
    tensor_y = torch.Tensor(spectra_RT)

    # generate torch dataset
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BTSZ, shuffle=True)

    return dataset, dataloader