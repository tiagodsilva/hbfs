data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  int<lower=0> M;   // number of players 
  matrix[N, K] x;   // predictor matrix
  array[N] int y;      // outcome vector
  array[N] int omega;  // player ids 
}
parameters {
  vector[M] alpha;      // fixed effects  
  vector[K] beta;       // coefficients for predictors
}
model {
  beta ~ normal(0., 1.); 
  alpha ~ normal(0. ,1.); 

  y ~ bernoulli_logit(x * beta + alpha[omega]);  // likelihood
}
