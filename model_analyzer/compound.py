import decimal
from math import exp, factorial

import numpy as np
import pandas as pd


"""
Compute alpha and beta required for computing the mean wait time (balking rate)
Returns a tuple in the order of alpha and beta
"""
def compute_alpha_beta(lambda_a:float, lambda_s:float, s:int, b:int) -> tuple:
  # delta short hand
  delta = s * lambda_s - lambda_a
  
  # Compute 'P'
  a = decimal.Decimal(lambda_a/lambda_s)
  P_num = (a)**(s)
  P_num = P_num / factorial(s)
  P_den = sum([((a**i)/factorial(i)) for i in range(s+1)])
  P = float(P_num / P_den)

  # Beta is inverse of expected value of idle hours
  beta = (s*lambda_s*P)/(1-P)

  # Effeciency
  rho = lambda_a/(s*lambda_s)

  # Alpha
  if rho  == 1:
    alpha = lambda_a/(lambda_a+beta*(1+lambda_a*b))
  else:
    alpha_inner = beta * ((1 / delta) - (np.exp(-delta * b) * (lambda_a/ (delta * s * lambda_s))))
    alpha = 1.0/(alpha_inner + 1)
  return alpha, beta

"""
Compute long job wait time for M/M/s queue
Returns a tuple in the order - mean wait time, and balking rate
"""
def compute_wait_time_mms(lambda_a:float, lambda_s: float, s:int, b: int) -> tuple:
  # Fraction of jobs running on on-demand (r)
  alpha, beta = compute_alpha_beta(lambda_a, lambda_s, s, b)
  delta = s * lambda_s - lambda_a
  r = (alpha*beta*np.exp(-1*delta*b))/(s*lambda_s)

  # Mean waiting time (w)
  rho = lambda_a/(s*lambda_s) # utilization
  if rho == 1:
    EW = alpha*beta*(b**2)/(2)
  else:
    EW = (alpha*beta*(1 - delta*b*np.exp(-delta*b)-np.exp(-delta*b)))/((delta**2))
  w = EW

  return (w, r)

"""
Given lambda, mu, s, and t (long job min threshold in sec). 
Compute lambda-prime and mu-prime i.e. new arrival rate and service rate for long jobs.
Returns a tuple in the order of lambda-prime and mu-prime
"""
def compute_new_rates(lambda_a: float, mu: float, t: int) -> tuple:
  # mu-prime
  mu_prime = mu / ((mu * t) + 1)

  long_job_perc = np.exp(-1 * mu * t) # fraction of long jobs (From CDF of service rate)

  # lambda_prime
  lambda_a_prime = lambda_a * long_job_perc

  # print ("Ratio of new lambda to new mu is ", lambda_a_prime / mu_prime)
  # print ("Long job perc is ", long_job_perc)
  return lambda_a_prime, mu_prime

"""
Computing coefficient of variation 
"""
def compute_cov(lambda_a: float, mu: float, t: int) -> float:
  new_lambda, new_mu = compute_new_rates(lambda_a, mu, t)
  new_service_mean = 1/new_mu
  std_dev = 1/(mu)
  cov = std_dev / new_service_mean
  return cov

"""
Compute mean wait time of LJW
"""
def compute_model_values(lambda_a: float, mu: float, p_f: float,
                           p_o: float, s: int, t: int, b: int) -> tuple:
  new_lambda, new_mu = compute_new_rates(lambda_a, mu, t)


  # fraction of long & short jobs (From CDF of service rate)
  long_job_perc = np.exp(-1 * mu * t)
  short_job_perc = 1 - long_job_perc

  # fraction of long & short job service times (From CDF of service rate)
  long_job_time_perc = long_job_perc * (mu / new_mu)
  short_job_time_perc = 1 - long_job_time_perc

  # Mean wait time (w)
  wt, r = compute_wait_time_mms(new_lambda, new_mu, s, b)
  cov = compute_cov(lambda_a, mu, t)
  approx_wt = wt * (((cov**2) + 1)/2)
  w = long_job_perc * approx_wt

  # Amortized price (p)
  rho = new_lambda / (s * new_mu)
  p_long = (p_f / rho) + r * p_o
  p_short = p_o
  p = long_job_time_perc * p_long + short_job_time_perc * p_short

  # Fraction of jobs running on on-demand jobs
  r_ond = short_job_perc + r * long_job_perc
  return (w, p, r_ond)

if __name__ == "__main__":
  lambda_a, mu = 0.2, 0.002
  t, b = 180, 900
  s = 90
  print (compute_model_values(lambda_a, mu, 3.84, 9.6, s, t, b))
