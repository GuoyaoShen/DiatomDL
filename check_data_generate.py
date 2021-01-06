import numpy as np

from utils.data_utils import generate_data

path_data_R = 'data/all silicon terahertz_325 sets/R_'
num_size_R = 13
num_param_R = 8

data_param_R, data_spectra_R = generate_data(path_data_R, num_size_R, num_param_R)

print(data_param_R.shape)
print(data_spectra_R.shape)




path_data_T = 'data/all silicon terahertz_325 sets/T_'
num_size_T = 13
num_param_T = 8

data_param_T, data_spectra_T = generate_data(path_data_T, num_size_T, num_param_T)

print(data_param_T.shape)
print(data_spectra_T.shape)

print(np.any(data_param_T - data_param_R))  # if this is false, then correct