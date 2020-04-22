"""
Copyright (c) 2020 <Sustainable Computing Lab>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import decimal
from math import exp, factorial

import numpy as np
import pandas as pd


"""
Given lambda, mu, s, and t (long job min threshold in sec). 
Compute new arrival rate and service rate for long jobs.
Returns a tuple in the order of lambda-prime and mu-prime
"""
def compute_new_rates(lambda_a: float, mu: float, t: int) -> tuple:
  # mu-prime
  mu_prime = mu / ((mu * t) + 1)

  # fraction of long jobs (From CDF of service rate)
  long_job_perc = np.exp(-1 * mu * t)

  # lambda_prime
  lambda_a_prime = lambda_a * long_job_perc

  return (lambda_a_prime, mu_prime)

"""
Computing coefficient of variation
Approximation for Long job waiting time
"""
def compute_cov(lambda_a: float, mu: float, t: int) -> float:
  new_lambda, new_mu = compute_new_rates(lambda_a, mu, t)
  new_service_mean = 1/new_mu
  std_dev = 1/(mu)
  cov = std_dev / new_service_mean
  return cov

"""
Erlang C formula
"""
def compute_erlang_c(lambda_a: float, mu: float, s: int) -> float:
  a = decimal.Decimal(lambda_a/mu)
  rho = a/s
  num = (a) ** (s)
  num  = num / (factorial(s) * (1-rho))
  den = sum([((a**i)/factorial(i)) for i in range(s)])
  den = den + num
  return float(num/den)

"""
Compute model values for LJW
"""
def compute_model_values(lambda_a:float, mu: float,
                          p_f: float, p_o: float, s:int, t: int) -> tuple:
  # Long job arrival and service rate
  new_lambda, new_mu = compute_new_rates(lambda_a, mu, t)

  # fraction of jobs running on on-demand (r)
  long_job_perc = np.exp(-1 * mu * t)
  r = 1 - long_job_perc 

  # Expected wait time (w)
  a = decimal.Decimal(new_lambda/new_mu) # offered load
  rho = float(a/s) # utilization
  erlang_c = compute_erlang_c(new_lambda, new_mu, s)
  EQ = erlang_c * (rho/(1-rho))  
  lw = EQ / new_lambda 
  cov = compute_cov(lambda_a, mu, t)
  approx_lw = lw * (((cov**2) + 1)/2) # Long job waiting time
  w = approx_lw * long_job_perc

  # fraction of long & short job service times (From CDF of service rate)
  long_job_time_perc = long_job_perc * (mu / new_mu)
  short_job_time_perc = 1 - long_job_time_perc
  
  # Amortized cost (p)
  rho = new_lambda / (s * new_mu) # Fixed resource utilization
  p_long = (p_f / rho)
  p_short = p_o
  p = p_long * long_job_time_perc + short_job_time_perc * p_short
  return (w, p, r)

if __name__ == "__main__":
  lambda_a, mu = 0.2, 0.002
  t = 180
  s = 96
  print (compute_model_values(lambda_a, mu, 3.84, 9.6, s, t))
