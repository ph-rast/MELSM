## Project title: A Mixed-Effects Location Scale Model for Dyadic Interactions
##    Created at: Jan. 25, 2018
##        Author: Philippe Rast & Emilio Ferrer
##          Data: One draw from predicted posterior outcome from original
##                data with highest likelihood.
## ---------------------------------------------------------------------- ##

## Datasets, preformatted for each model
load(file = "MELSM.rda")
ls()

## Peak into data:
str(dat200_m3)

## List of 10
##  $ group: num [1:8451] 1 1 1 1 1 1 1 1 1 1 ...          ## Subject number 1-200
##  $ y    : num [1:8451] 1.94 1.7 4.45 3.46 3.28 ...      ## Dependent variable (PA affect)
##  $ w    :'data.frame':	8451 obs. of  4 variables: ## Design matrix for scale
##   ..$ Females       : num [1:8451] 1 1 1 1 1 1 1 1 1 1 ... ## Female
##   ..$ FemXNApartner : num [1:8451] 0.254 -0.186 -0.186 ... ## Interaction term w. partner's NA
##   ..$ Males         : num [1:8451] 0 0 0 0 0 0 0 0 0 0 ... ## Male
##   ..$ MaleXNApartner: num [1:8451] 0 0 0 0 0 0 0 0 0 0 ... ## Interaction term
##  $ x    :'data.frame':	8451 obs. of  6 variables: ## Design matrix for location
##   ..$ Females       : num [1:8451] 1 1 1 1 1 1 1 1 1 1 ... ## Female 
##   ..$ FemXPAlag     : num [1:8451] -0.862 -0.412 1.138 ... ## Interaction term with lag1
##   ..$ FemXPApartner : num [1:8451] 0.1855 -0.0345 1.29 ... ## Interaction w. partner's PA
##   ..$ Males         : num [1:8451] 0 0 0 0 0 0 0 0 0 0 ... ## Male
##   ..$ MaleXPAlag    : num [1:8451] 0 0 0 0 0 0 0 0 0 0 ... ## Interaction term with lag1
##   ..$ MaleCPApartner: num [1:8451] 0 0 0 0 0 0 0 0 0 0 ... ## Interaction w. partner's PA
##  $ nobs : int 8451                                     ## Number of observations
##  $ J    : num 200                                      ## Number of subjects
##  $ z    : num [1:200, 1:3] 1 1 1 1 1 1 1 1 1 1 ...     ## Location: Between-Person predictors at level 2
##  $ m    : num [1:200, 1:3] 1 1 1 1 1 1 1 1 1 1 ...     ## Scale: Between-Person predictors at level 2
##  $ g    : num [1:200, 1] 1 1 1 1 1 1 1 1 1 1 ...       ## Location random effects predictors
##  $ a    : num [1:200, 1:2] 1 1 1 1 1 1 1 1 1 1 ...     ## Scale random effect predictors

############################################
## load Stan and loo for model comparison ##
############################################
library('rstan')
## enable multicore computing
rstan_options(auto_write = TRUE) 
options(mc.cores = parallel::detectCores())
library(loo)

################################
## Basic Mixed Effects Model  ##
################################

model_mixed <- '
data {
  int<lower=0> nobs;                // num of observations
  int<lower=1> J;                   // number of groups or subjects
  int<lower=1,upper=J> group[nobs]; // vector with group ID
  matrix[nobs,6] x;                 // design matrix w. time-varying wp predictors for location
  matrix[nobs,2] w;                 // design matrix w. time-varying wp predictors for scale
  matrix[J,3] z;                    // between person predictors at level 2 for location
  vector<lower=1, upper=5>[nobs] y; // column vector with outcomes
}
// Parameters to be estimated
parameters {                        
  cholesky_factor_corr[6] L_Omega;// Cholesky decomposition of Omega
  matrix[6,3] gamma;              // Fixed effects
  matrix[6,J] stdnorm;            // Standard normal, used to multiply w. cholesky factor to obtain multivariate normal beta
  vector<lower=0>[6] tau;         // Vector of random effect SDs
  vector[2] log_sigma;            // Residual SD on log-scale
}
transformed parameters {
  matrix[J,6] mu;
  matrix[J,6] beta;
  vector<lower=0>[nobs] sigma;
  // Level 2
  mu = z * transpose(gamma);
  sigma = exp(w * log_sigma);
  beta = mu + transpose(diag_pre_multiply(tau, L_Omega)*stdnorm);
}
model {
// Priors
  tau ~ cauchy(0, 2);
  to_vector(stdnorm) ~ normal(0,1);
  L_Omega ~ lkj_corr_cholesky(1);
  to_vector(gamma) ~ normal(0, 100);
// likelihood
  y ~ normal(rows_dot_product(beta[group, 1:6], x), sigma);
}
generated quantities {  // This section is not necessary, but contains useful transformations and generates data for posterior checks     
  corr_matrix[6] Omega;              // Obtain Omega from Cholesky factor to print in output
  vector[2] Sigma;                   // Antilog of log_sigma
  vector[nobs] log_lik;
  Omega = L_Omega*transpose(L_Omega);// Correlation matrix for output
  Sigma = exp(log_sigma);
  for (i in 1:nobs){                 // log-likelihoods to compute loo
   log_lik[i] = normal_lpdf(y[i] |x[i,1]*beta[group[i],1] + x[i,2]*beta[group[i],2] + x[i,3]*beta[group[i],3] +
                                  x[i,4]*beta[group[i],4] + x[i,5]*beta[group[i],5] + x[i,6]*beta[group[i],6] ,
                                  sigma[i]);
   }
}'

## Run model
mixed <- stan(model_code = model_mixed, data = dat200_m0, verbose = TRUE, iter = 500) 

## Print summary statistics of posteriors
print(mixed, pars=c('Omega', 'gamma','tau','Sigma','lp__'), probs=c(.025, .975), digits = 2)

## Predictive performance measures
log_lik_mixed <- extract_log_lik(mixed)
loo_mixed  <- loo(log_lik_mixed)
waic_mixed <- waic(log_lik_mixed)
print(loo_mixed)
print(waic_mixed)

## Free up memory
rm(mixed)
rm(log_lik_mixed)

##############################################################################################
## MELSM 1:                                                                                 ##
##                                                                                          ##
## Random effects of scale (intercept only) are now correlated with location random effects ##
##############################################################################################
model_1 <- '
data {
  int<lower=0> nobs;      // num of observations
  int<lower=1> J;         // number of groups or subjects
  int<lower=1,upper=J> group[nobs]; // vector with group ID
  matrix[nobs,6] x;       // design matrix w. time-varying wp predictors for location
  matrix[nobs,2] w;       // design matrix w. time-varying wp predictors for scale
  matrix[J,3] z;          // between person predictors at level 2 for location
  matrix[J,2] m;          // between person predictors at level 2 for scale
  vector<lower=1, upper=5>[nobs] y; // column vector with outcomes
}
// Parameters to be estimated
parameters {                        
  cholesky_factor_corr[8] L_Omega;// Cholesky decomposition of Omega
  matrix[6,3] gamma;              // Location fixed effects
  matrix[2,2] xi;                 // Scale fixed effects
  matrix[8,J] stdnorm;            // Standard normal, multiply w. cholesky factor to obtain multivariate normal beta
  vector<lower=0>[8] tau;         // Vector of random effect SDs
}
transformed parameters {
  matrix[J,6] z_gamma;
  matrix[J,2] m_xi;
  matrix[J,8] mu;
  matrix[J,8] beta;
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
         exp(rows_dot_product(beta[group, 7:8], w)));
}
generated quantities { 
  corr_matrix[8] Omega;  // Obtain Omega from Cholesky factor.
  vector[nobs] log_lik;
  Omega = L_Omega*transpose(L_Omega);
  for (i in 1:nobs){
   log_lik[i] = normal_lpdf(y[i] |x[i,1]*beta[group[i],1] + x[i,2]*beta[group[i],2] + x[i,3]*beta[group[i],3] +
                                  x[i,4]*beta[group[i],4] + x[i,5]*beta[group[i],5] + x[i,6]*beta[group[i],6] ,
                              exp(w[i,1]*beta[group[i],7] +
                                  w[i,2]*beta[group[i],8]  ));
   }
}'

## Run model
melsm1 <- stan(model_code = model_1, data = dat200_m1, verbose = TRUE, iter = 500) 

print(melsm1, pars=c('Omega', 'gamma','xi', 'tau','lp__'), probs=c(.025, .975), digits = 2)

log_lik_1 <- extract_log_lik(melsm1)
loo_1  <- loo(log_lik_1)
waic_1 <- waic(log_lik_1)
print(loo_1)
print(waic_1)


print(loo_mixed)
print(loo_1)
compare(loo_mixed, loo_1)
## large positve difference with smallish standard error
## loo_1 is preferred model

## Free up memory
rm(log_lik_1)
rm(melsm1)

##############################################################################################
## MELSM 2:                                                                                 ##
##                                                                                          ##
## Random effects of scale (intc and slope) are now correlated with location random effects ##
##############################################################################################

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
  cholesky_factor_corr[10] L_Omega;// Cholesky decomposition of Omega
  matrix[6,3] gamma;               // Location fixed effects
  matrix[4,3] xi;                  // Scale fixed effects
  matrix[10,J] stdnorm;            // Standard normal, multiply w. cholesky factor to obtain multivariate normal beta
  vector<lower=0>[10] tau;         // Vector of random effect SDs
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
// This section is not necessary, but contains useful transformations and generates data for posterior checks   
  corr_matrix[10] Omega;   // Obtain Omega from Cholesky factor to print in output
  vector[nobs] log_lik;
  vector[nobs] y_rep;
  Omega = L_Omega*transpose(L_Omega); // Correlation matrix for output
  for (i in 1:nobs){                  // Log-likelihoods and predicted y for loo and posterior predictive checks
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
melsm2 <- stan(model_code = model_2, data = dat200_m2, verbose = TRUE, iter = 800) 

print(melsm2, pars=c('Omega', 'gamma','xi', 'tau','lp__'), probs=c(.025, .975), digits = 2)

## Compare empirical distribution of data 'y' to the distribution of replicated data 'yrep'
yrep <- extract(melsm2)[['y_rep']]  ## replicated data

plot(density(dat200_m2$y), type = 'n')
for(i in sample(1:nrow(yrep), 200)){
    lines(density(yrep[i,]), col = 'gray80')
}
lines(density(dat200_m2$y), col = 'red', lwd = 3)


log_lik_2 <- extract_log_lik(melsm2)
loo_2  <- loo(log_lik_2)
waic_2 <- waic(log_lik_2)
print(loo_2)
print(waic_2)

compare(loo_1, loo_2) ## difference suggests that loo_2 has better predictive performance

## Free up memory
rm(yrep)
rm(log_lik_2)
rm(melsm2)

##############################################################################################
## MELSM 3:                                                                                 ##
##                                                                                          ##
## Random effects of scale (intc and slope) are now correlated with location random effects ##
## as well as variance model for scale random effects                                       ## 
##############################################################################################
model_3<-'
data {
  int<lower=0> nobs;   // number of observations
  int<lower=1> J;      // number of groups or subjects
  int<lower=1,upper=J> group[nobs]; // vector with group ID
  matrix[nobs,6] x;   // design matrix w. time-varying wp predictors for location
  matrix[nobs,4] w;   // design matrix w. time-varying wp predictors for scale
  matrix[J,3] z;      // between person predictors at level 2 for location
  matrix[J,3] m;      // between person predictors at level 2 for scale
  matrix[J,1] g;      // between person predictors for location ranefvar (intercept only)
  matrix[J,2] a;      // between person predictors for scale ranefvar (intercept and slope)
  vector<lower=1,upper=5>[nobs] y; // column vector with outcomes
}
parameters {
  cholesky_factor_corr[10] L_Omega; // Cholesky decomposition of Omega
  matrix[6,3] gamma;               // Location Fixed effects
  matrix[4,3] xi;                  // Scale fixed effects
  matrix[6,1] iota_l;              // iota, SD, for location random effects
  matrix[4,2] iota_s;              // iota, SD, for scale random effects (modeled with predictors in a)
  matrix[10,J] stdnorm;            // Standard normal, used to multiply w. cholesky factor to obtain multivariate normal beta
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
  g_iota_l = exp(g * transpose(iota_l));  // submodel for location random effect SDs (intercept only)
  a_iota_s = exp(a * transpose(iota_s));  // submodel for scale random effect SDs (intercept and slope)
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

## Visual check of convergence (for e.g. gamma[1,1])
traceplot(melsm3, pars = c('gamma[1,1]'), inc_warmup = TRUE)

## Compare to model 2 via loo
log_lik_3 <- extract_log_lik(melsm3)
loo_3  <- loo(log_lik_3)

compare(loo_2, loo_3) # small difference in predictive performance among MELSM2 and MELSM3

## Compare empirical distribution of data 'y' to the distribution of replicated data 'yrep'
yrep <- extract(melsm3)[['y_rep']]  ## replicated data

plot(density(dat200_m3$y), type = 'n')
for(i in sample(1:nrow(yrep), 200)){
    lines(density(yrep[i,]), col = 'gray80')
}
lines(density(dat200_m2$y), col = 'red', lwd = 3)
