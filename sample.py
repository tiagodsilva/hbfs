import stan 
import numpy as np 
import json 
if __name__ == '__main__': 
    stan_code = open('model.stan', 'r').read() 

    # Generate data 
    mtype = 'random' 
    data = json.load(open(f'data{mtype}.json')) 

    # Cast the data to the correct types 
    data['omega'] = np.array(data['omega']).astype(int) 
    data['y'] = np.array(data['y']).astype(int) 
        
    posterior = stan.build(stan_code, data=data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=2000) 

    fit.to_frame().to_csv('data.csv')  