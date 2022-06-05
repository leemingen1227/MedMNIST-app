import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


npz_file = np.load('./test_image/pathmnist.npz')

x = npz_file['test_images']
y = npz_file['test_labels']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.9, random_state=0)

np.savez('pathmnist_test.npz', test_images=X_train, test_labels=y_train)
