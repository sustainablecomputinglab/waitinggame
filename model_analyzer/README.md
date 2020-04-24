# Model Analyzer

Our model analyzer implements the analytical queuing model for all the waiting policies we analyze in the paper. The analyzer enables enables what-if analyses to compare and understand a workload’s expected cost and job waiting times under different policies and parameter values.

## Usage

*1.* The analyzer (model_analyzer.py) takes as input (command line arguments) a policy’s name and λ, µ, s, p_f, and p_o, as well as b for AJW-T, SWW, and the compound policy, and t for LJW and the compound policy.

```python
# Shows usage
python model_analyzer --h

# NJW
python model_analyzer.py --policy NJW --lmbda 0.2 --mu 0.002 --s 108 --pf 3.84 --po 9.6
```

*2.*  The analyzer’s output is the policy’s mean waiting time w, the effective price P, the fraction of jobs that run on on-demand resources r.

```python
# Example
$ python model_analyzer.py --policy NJW --lmbda 0.2 --mu 0.002 --s 108 --pf 3.84 --po 9.6
Mean waiting time in seconds -  0 , effective price is  4.482662486093569 , and the fraction of jobs that run on on-demand resources is  0.03494400896808022
```
