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
Erlang C formula
"""
def compute_erlang_c(lambda_a: float, lambda_s: float, s: int) -> float:
  a = decimal.Decimal(lambda_a/lambda_s)
  rho = a/s
  num = (a) ** (s)
  num  = num / (factorial(s) * (1-rho))
  den = sum([((a**i)/factorial(i)) for i in range(s)])
  den = den + num
  return float(num/den)

"""
Compute model values for AJW
"""
def compute_model_values(lambda_a:float, lambda_s: float,
                           p_f: float, p_o: float, s:int) -> tuple:
  a = decimal.Decimal(lambda_a/lambda_s) # offered load
  rho = float(a/s) # utilization
  p = p_f / rho # amortized price

  # mean waiting time
  erlang_c = compute_erlang_c(lambda_a, lambda_s, s)
  EQ = erlang_c * (rho/(1-rho))  
  w = EQ / lambda_a
  
  # fraction of jobs running on on-demand
  r = 0
  return (w, p, r)

if __name__ == "__main__":
  lambda_a = 0.2
  lambda_s = 0.002
  s = 101
  p_o = 9.6
  p_f = 0.4 * p_o
  print (compute_model_values(lambda_a, lambda_s, p_f, p_o, s))
