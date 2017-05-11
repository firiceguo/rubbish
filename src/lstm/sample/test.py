import numpy as np
import os

now_path = os.getcwd() + '/'
f = np.load(now_path + 'imdb.npz')

print f.files
