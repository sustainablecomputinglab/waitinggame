import argparse
import njw, ajw, ajw_t, sww, ljw, compound

# Parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--policy", help="Policy Name", type=str)
parser.add_argument("--lmbda", help="Arrival Rate", type=float)
parser.add_argument("--mu", help="Service Rate", type=float)
parser.add_argument("--s", help="Number of Servers", type=int)
parser.add_argument("--pf", help="Fixed Price", type=float)
parser.add_argument("--po", help="On-demand Price", type=float)
parser.add_argument("--b", help="Maximum Wait Time in Seconds", nargs='?', type=int)
parser.add_argument("--t", help="Maximum Wait Time in Seconds", nargs='?', type=int)

# Read Command Line arguments
args = parser.parse_args()

# Check policy
policies = ['NJW', 'AJW', 'AJW-T', 'SWW', 'LJW', 'Compound']
if args.policy not in policies:
  print ("Please select one of following policies - ", policies)
  exit()

# NJW
if args.policy == 'NJW':
  w, p ,r = njw.compute_model_values(args.lmbda, args.mu, args.pf, args.po, args.s)
  print ("Mean waiting time in seconds - ", w, ", effective price is ", p,
         ", and the fraction of jobs that run on on-demand resources is ", r)

# AJW
if args.policy == 'AJW':
  w, p ,r = ajw.compute_model_values(args.lmbda, args.mu, args.pf, args.po, args.s)
  print ("Mean waiting time in seconds - ", w, ", effective price is ", p,
         ", and the fraction of jobs that run on on-demand resources is ", r)

# AJW-T
if args.policy == 'AJW-T':
  if args.b is None:
    print ("Please specify the maximum waiting time (b) for AJW-T policy.")
    exit()
  w, p ,r = ajw_t.compute_model_values(args.lmbda, args.mu, args.pf, args.po, args.s, args.b)
  print ("Mean waiting time in seconds - ", w, ", effective price is ", p,
         ", and the fraction of jobs that run on on-demand resources is ", r)

# SWW
if args.policy == 'SWW':
  if args.b is None:
    print ("Please specify the maximum waiting time (b) for SWW policy.")
    exit()
  w, p ,r = sww.compute_model_values(args.lmbda, args.mu, args.pf, args.po, args.s, args.b)
  print ("Mean waiting time in seconds - ", w, ", effective price is ", p,
         ", and the fraction of jobs that run on on-demand resources is ", r)

# LJW
if args.policy == 'LJW':
  if args.t is None:
    print ("Please specify the long job threshold (t) for LJW policy.")
    exit()
  w, p ,r = ljw.compute_model_values(args.lmbda, args.mu, args.pf, args.po, args.s, args.t)
  print ("Mean waiting time in seconds - ", w, ", effective price is ", p,
         ", and the fraction of jobs that run on on-demand resources is ", r)

# Compound
if args.policy == 'Compound':
  if args.t is None or args.b is None:
    print ("Please specify both long job threshold (t) and maximum waiting time (b) for Compound policy.")
    exit()
  w, p ,r = compound.compute_model_values(args.lmbda, args.mu, args.pf, args.po, args.s, args.t, args.b)
  print ("Mean waiting time in seconds - ", w, ", effective price is ", p,
         ", and the fraction of jobs that run on on-demand resources is ", r)