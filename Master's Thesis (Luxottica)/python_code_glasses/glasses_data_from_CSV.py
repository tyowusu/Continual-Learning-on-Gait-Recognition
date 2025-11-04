import pandas as pd
from glasses_utils import *
import os

plot = True
save = True

directory = 'dataset_luxottica/original/'

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        print(filepath)

        fast_df, slow_df, interval = load_data(filepath)

        if save:
            fast_df.to_csv('dataset_luxottica/fast/' + filename[:-4] + '_fast.csv')
            slow_df.to_csv('dataset_luxottica/slow/' + filename[:-4] + '_slow.csv')
            with open('dataset_luxottica/intervals/' + filename[:-4] + '_interval.txt', 'w') as f:
                f.write(interval)

        if plot: 
            plot_data(fast_df, slow_df, filepath)
