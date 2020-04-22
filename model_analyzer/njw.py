import decimal
from math import exp, factorial

import numpy as np
import pandas as pd


# Erlang B formula
def compute_blocking_probab(lambda_a: float, lambda_s: float, s: int) -> float:
  x = decimal.Decimal(lambda_a / lambda_s)
  num = (x) ** (s)
  num = num / factorial(s)
  den = sum([((x**i)/factorial(i)) for i in range(s+1)])
  return float(num/den)

# Amortized Cost
def compute_model_values(lambda_a:float, lambda_s: float,
                           p_f: float, p_o: float, s:int) -> tuple:
  r = compute_blocking_probab(lambda_a, lambda_s, s) # Blocking probability (r)
  rho = lambda_a/(s*lambda_s) # Utilization
  p = (p_f/rho) + r * p_o # Amortized price
  w = 0 # Mean waiting time
  return (w,p,r)


if __name__ == "__main__":
  lambda_s = 0.002
  lambda_a = 0.2
  w, p, r = compute_model_values(lambda_a, lambda_s, 3.84, 9.6, 108)
  print (w, p, r)