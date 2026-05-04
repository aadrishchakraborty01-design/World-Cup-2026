import os
import shutil
import time
import pandas as pd
import numpy as np
import kagglehub
from tqdm import tqdm
RAW = '../data/raw'
PROC = '../data/processed'
os.makedirs('../data/raw/statsbomb', exist_ok = True)
os.makedirs('../data/processed', exist_ok = True)
print("Cell 1 ready")


path = kagglehub.dataset_download(martj42/international-football-results-from-1872-to-2017)
for f in oslistdir(path):
    dest = 