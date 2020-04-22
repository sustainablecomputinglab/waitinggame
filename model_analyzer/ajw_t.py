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
Compute model values for AJW-T
Returns a tuple in the order - mean wait time, amortized price, and balking rate
"""
def compute_model_values(lambda_a:float, lambda_s: float,
                          p_f: float, p_o: float, s:int, b: int) -> tuple:
  # Fraction of jobs running on on-demand (r)
  alpha, beta = compute_alpha_beta(lambda_a, lambda_s, s, b)
  delta = s * lambda_s - lambda_a
  r = (alpha*beta*np.exp(-1*delta*b))/(s*lambda_s)

  # Amortized cost (p)
  rho = lambda_a/(s*lambda_s) # utilization
  p = (p_f/rho) + r * p_o 

  # Mean waiting time (w)
  alpha, beta = compute_alpha_beta(lambda_a, lambda_s, s, b)
  delta = s * lambda_s - lambda_a
  if rho == 1:
    EW = alpha*beta*(b**2)/(2)
  else:
    EW = (alpha*beta*(1 - delta*b*np.exp(-delta*b)-np.exp(-delta*b)))/((delta**2))
  w = EW + (r * b)

  return (w, p, r)

if __name__ == "__main__":
  lambda_a = 0.2
  lambda_s = 0.002
  s = 101
  p_o = 9.6
  p_f = 0.4 * p_o
  b = 900
  print (compute_model_values(lambda_a, lambda_s, p_f, p_o, s, b))
