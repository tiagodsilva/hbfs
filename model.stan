data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  int<lower=0> M;   // number of players 
  matrix[N, K] x;   // predictor matrix
  vector[N] y;      // outcome vector
  array[N] int omega;  // player id  
}
parameters {
  vector[M] alpha;      // fixed effects  
  vector[K] beta;       // coefficients for predictors
}
model {
  beta ~ normal(0., 1.); 
  alpha ~ normal(0. ,1.); 

  y ~ bernoulli(logit(x * beta + alpha[omega]));  // likelihood
}
