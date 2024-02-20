import numpy as np 
import pandas as pd 
import json 
import matplotlib.pyplot as plt 
import sys 

from itertools import combinations 

sys.path.append('.') 

from simulate_data import mtype 


if __name__ == '__main__': 
    metadata = json.load(open(f'data{mtype}.json', 'r')) 
    num_players = metadata['M'] 

    samples_from_hmc = pd.read_csv('data.csv') 

    stats = list() 

    for i, j in combinations(range(1, num_players + 1), 2):
        alpha_i = samples_from_hmc[f'alpha.{i}'] 
        alpha_j = samples_from_hmc[f'alpha.{j}'] 

        p = (alpha_i >= alpha_j).mean() 
        stats.append(p) 

    plt.hist(stats) 
    plt.savefig('fig.pdf') 

