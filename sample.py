import stan 
import numpy as np 

if __name__ == '__main__': 
    stan_code = open('model.stan', 'r').read() 

    # Generate data 
    data = {
        "N": 12, 
        "K": 12, 
        "M": 3, 
        "x": np.random.randn(12, 12), 
        "y": np.random.randn(12), 
        "omega": np.random.randint(2, size=(12,)) + 1 
    }
    
    posterior = stan.build(stan_code, data=data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=2000) 

    fit.to_frame().to_csv('data.csv')  