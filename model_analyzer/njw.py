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