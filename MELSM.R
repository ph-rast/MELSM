## Project title: A Mixed-Effects Location Scale Model for Dyadic Interactions
##    Created at: Jan. 25, 2018
##        Author: Philippe Rast & Emilio Ferrer
##          Data: One draw from predicted posterior outcome from original
##                data with highest likelihood.
## ---------------------------------------------------------------------- ##

## Two datasets for model 2 and model 3
load(file = "MELSM.RData")

#############################
## load Stan
#############################
library('rstan')
## enable multicore computing
rstan_options(auto_write = TRUE) 
options(mc.cores = parallel::detectCores())


model_2 <- '
data {
  int<lower=0> nobs;      // num of observations
  int<lower=1> J;         // number of groups or subjects
  int<lower=1,upper=J> group[nobs]; // vector with group ID
  matrix[nobs,6] x;       // design matrix w. time-varying wp predictors for location
  matrix[nobs,4] w;       // design matrix w. time-varying wp predictors for scale
  matrix[J,3] z;          // between person predictors at level 2 for location
  matrix[J,3] m;          // between person predictors at level 2 for scale
  vector<lower=1, upper=5>[nobs] y; // column vector with outcomes
}
// Parameters to be estimated
parameters {                        
  cholesky_factor_corr[10] L_Omega; // Cholesky decomposition of Omega
  matrix[6,3] gamma;
  matrix[4,3] xi;
  matrix[10,J] stdnorm;
  vector<lower=0>[10] tau;
}
transformed parameters {
  matrix[J,6] z_gamma;
  matrix[J,4] m_xi;
  matrix[J,10] mu;
  matrix[J,10] beta;
  // Level 2
  z_gamma = z * transpose(gamma);
  m_xi = m * transpose(xi);
  mu = append_col(z_gamma, m_xi);
  beta = mu + transpose(diag_pre_multiply(tau, L_Omega)*stdnorm);
}
model {
// Priors
  tau ~ cauchy(0, 2);
  to_vector(stdnorm) ~ normal(0,1);
  L_Omega ~ lkj_corr_cholesky(1);
  to_vector(xi) ~ normal(0, 100);
  to_vector(gamma) ~ normal(0, 100);
// likelihood
  y ~ normal(rows_dot_product(beta[group, 1:6], x),
         exp(rows_dot_product(beta[group, 7:10], w)));
}
generated quantities { 
  corr_matrix[10] Omega;  // Obtain Omega from Cholesky factor.
  vector[nobs] log_lik;
  vector[nobs] y_rep;
  Omega = L_Omega*transpose(L_Omega);
  for (i in 1:nobs){
   log_lik[i] = normal_lpdf(y[i] |x[i,1]*beta[group[i],1] + x[i,2]*beta[group[i],2] + x[i,3]*beta[group[i],3] +
                                  x[i,4]*beta[group[i],4] + x[i,5]*beta[group[i],5] + x[i,6]*beta[group[i],6] ,
                              exp(w[i,1]*beta[group[i],7] + w[i,2]*beta[group[i], 8] +
                                  w[i,3]*beta[group[i],9] + w[i,4]*beta[group[i],10] ));
     y_rep[i] = normal_rng(x[i,1]*beta[group[i],1] + x[i,2]*beta[group[i],2] + x[i,3]*beta[group[i],3] +                                                
                           x[i,4]*beta[group[i],4] + x[i,5]*beta[group[i],5] + x[i,6]*beta[group[i],6] ,
                       exp(w[i,1]*beta[group[i],7] + w[i,2]*beta[group[i], 8] +
                           w[i,3]*beta[group[i],9] + w[i,4]*beta[group[i],10] ));  
   }
}'

## Run model
melsm2 <- stan(model_code = model_2, data = dat200_m2, verbose = TRUE, iter = 1000) 

print(melsm2, pars=c('Omega', 'gamma','xi', 'tau','lp__'), probs=c(.025, .975), digits = 2)

## Compare empirical distribution of data 'y' to the distribution of replicated data 'yrep'
yrep <- extract(melsm2)[['y_rep']]  ## replicated data

plot(density(dat200_m2$y), type = 'n')
for(i in sample(1:nrow(yrep), 200)){
    lines(density(yrep[i,]), col = 'gray80')
}
lines(density(dat200_m2$y), col = 'red', lwd = 3)


library(loo)

log_lik_2 <- extract_log_lik(melsm2)
loo_2  <- loo(log_lik_2)
waic_2 <- waic(log_lik_2)
print(loo_2)
print(waic_2)

## Free up memory
rm(log_lik_2)
rm(melsm2)

## Location Scale model (MELSM3) with variance model for random effects
model_3<-'
data {
  int<lower=0> nobs;   // number of observations
  int<lower=1> J;      // number of groups or subjects
  int<lower=1,upper=J> group[nobs]; // vector with group ID
  matrix[nobs,6] x;   // design matrix w. time-varying wp predictors for location
  matrix[nobs,4] w;   // design matrix w. time-varying wp predictors for scale
  matrix[J,3] z;      // between person predictors at level 2 for location
  matrix[J,3] m;      // between person predictors at level 2 for scale
  matrix[J,1] g;      // between person predictors at for location ranefvar
  matrix[J,2] a;      // between person predictors at for scale ranefvar
  vector<lower=1,upper=5>[nobs] y; // column vector with outcomes
}
parameters {
  cholesky_factor_corr[10] L_Omega;
  matrix[6,3] gamma;
  matrix[4,3] xi;
  matrix[6,1] iota_l;   // iota for location
  matrix[4,2] iota_s;   // iota for scale
  matrix[10,J] stdnorm;
}
transformed parameters {
  matrix[J, 6] z_gamma;
  matrix[J, 4] m_xi;
  matrix[J,10] mu;
  matrix[J, 6] g_iota_l;
  matrix[J, 4] a_iota_s;
  matrix[J,10] tau;
  matrix[J,10] beta;
  z_gamma = z * transpose(gamma);
  m_xi = m * transpose(xi);
  mu = append_col(z_gamma, m_xi);
  g_iota_l = exp(g * transpose(iota_l));
  a_iota_s = exp(a * transpose(iota_s));
  tau = append_col(g_iota_l, a_iota_s);
  for(j in 1:J){
    beta[j,] = mu[j,] + transpose(diag_pre_multiply(tau[j,], L_Omega)*stdnorm[,j]);
  }
}
model {
// priors
  to_vector(stdnorm) ~ normal(0,1);
  L_Omega ~ lkj_corr_cholesky(1.5);
  to_vector(gamma) ~ normal(0.1, 0.5);
  gamma[1,1] ~ normal(3.45, 0.5); // intercepts obtain mean from MELSM 2
  gamma[4,1] ~ normal(3.60, 0.5);
  to_vector(xi) ~ normal(-0.5, 0.5);
  to_vector(iota_l) ~ normal(-1.5, 3);
  to_vector(iota_s) ~ normal(-1.5, 3);
// likelihood
  y ~ normal(rows_dot_product(beta[group, 1:6], x),
         exp(rows_dot_product(beta[group, 7:10], w)));
}
generated quantities { 
  corr_matrix[10] Omega;  // Obtain Omega from Cholesky factor.
  vector[nobs] log_lik;
  vector[nobs] y_rep;
  Omega = L_Omega*transpose(L_Omega);
  for (i in 1:nobs){
   log_lik[i] = normal_lpdf(y[i] |x[i,1]*beta[group[i],1] + x[i,2]*beta[group[i],2] + x[i,3]*beta[group[i],3] +
                                  x[i,4]*beta[group[i],4] + x[i,5]*beta[group[i],5] + x[i,6]*beta[group[i],6] ,
                              exp(w[i,1]*beta[group[i],7] + w[i,2]*beta[group[i], 8] +
                                  w[i,3]*beta[group[i],9] + w[i,4]*beta[group[i],10] ));
     y_rep[i] = normal_rng(x[i,1]*beta[group[i],1] + x[i,2]*beta[group[i],2] + x[i,3]*beta[group[i],3] +                                                
                           x[i,4]*beta[group[i],4] + x[i,5]*beta[group[i],5] + x[i,6]*beta[group[i],6] ,
                       exp(w[i,1]*beta[group[i],7] + w[i,2]*beta[group[i], 8] +
                           w[i,3]*beta[group[i],9] + w[i,4]*beta[group[i],10] ));  
   }
}'

## Run model with dat200_m3 which includes the predictor for the scale random effects variance
melsm3 <- stan(model_code = model_3, data = dat200_m3, verbose = TRUE, iter = 2000) 

print(melsm3, pars=c('Omega', 'gamma','xi', 'iota_l', 'iota_s','lp__'), probs=c(.025, .975), digits = 2)

## Compare to model 2 via loo
log_lik_3 <- extract_log_lik(melsm3)
loo_3  <- loo(log_lik_3)

compare(loo_2, loo_3)
