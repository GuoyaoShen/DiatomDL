import numpy as np

from utils.data_utils import generate_dataset


PATH_ZIPSET = 'data/allsilicone_325.npz'
dataset, dataloader = generate_dataset(PATH_ZIPSET)

print(len(dataset))
print(len(dataloader))