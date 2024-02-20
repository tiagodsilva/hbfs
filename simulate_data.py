import numpy as np 
import json 
import torch 

np.random.seed(42) 

def simulate_params(dim: int, num_players: int, alpha_type: str = 'linear'): 
    match alpha_type: 
        case 'linear': 
            alpha = np.linspace(0, num_players, num=num_players) 
            alpha -= num_players / 2
            alpha /= (num_players/4) 
        case 'random': 
            indices = np.random.choice(2, size=(num_players,), replace=True) 
            alpha = indices * (np.random.randn(num_players) - 2) + (1 - indices) * (np.random.randn(num_players) + 2) 
        case _: 
            raise Exception 
    
    # alpha, beta 
    return alpha, np.random.randn(dim) 

def simulate_data(num_samples: int, dim: int, num_players: int, alpha_type: str = 'linear'): 
    x = np.random.randn(dim)  
    x = np.repeat(x, repeats=num_samples).reshape(dim, num_samples).T  
    alpha, beta = simulate_params(dim, num_players, alpha_type) 

    samples_per_player = num_samples // num_players 
    omega = list() 
    ys = list() 

    for p in range(num_players): 
        logits = x[p*samples_per_player:(p+1)*samples_per_player] @ beta + alpha[p] 
        dist = torch.distributions.Bernoulli(logits=torch.tensor(logits)) 
        ys.append(
            dist.sample(sample_shape=(1,)).numpy().squeeze()
        ) 
        omega.append(np.ones((samples_per_player,)) * (p + 1)) 

    omega = np.hstack(omega) 
    ys = np.hstack(ys) 
    
    return x, omega, ys 
 
if __name__ == '__main__': 
    num_samples = int(3e3) 
    dim = 2 
    num_players = 150 
    mtype = 'random' 

    x, omega, ys = simulate_data(num_samples, dim, num_players, mtype)
    
    json.dump(
        {'N': num_samples, 'K': dim, 'M': num_players, 
         'x': x.tolist(), 'y': ys.tolist(), 'omega': omega.tolist()}, 
        open(f'data{mtype}.json', 'w') 
    )
    
