# Implementation of event based simulator

We implemented a trace-driven job simulator in python that mimics a cloud-enabled job scheduler, which can acquire VMs on-demand to service jobs. The simulator uses a FCFS scheduling policy, and also implements each of our waiting policies. We have implemented NJW (no jobs waiting), AJW (all jobs waiting), AJW-T (all jobs waiting threshold), SWW (short waits wait), LJW (Long jobs waiting), and Compound policy.

## To run the simulator (simulator.py)

*1.* Inputs to the simulator are input dataframe (input data), reserve capacity, year (dummy), cpu per reserve machine, and memory per reserve machine. Note that, all reserve VMs have to be of same type.

```python
year=2016
input_location = "Location of trace"
input_df = pd.read_hdf(input_location) # Assumes input as hdf.
s = 108 # number of servers
cpu = 1 # number of cores per server
mem_gb = 4 # Memory in GB per Server
sim = Simulator(year, s, cpu, mem_gb, input_trace)
```

*2.* Once the simulator instance is created, simulator can run simulations with different waiting policies. We have implemented 6 different waiting policies. Please check the following examples to simulate each of those waiting policies.

```Python
# NJW
sim.run_NJW()

# AJW
sim.run_AJWT(max_wait_time_min=sys.maxsize, SWW=False)

# AJW-T
b = 15 # max wait time in min
sim.run_AJWT(max_wait_time_min=b, SWW=False)

# SWW
b = 15 # maximum wait time in minutes
sim.run_AJWT(max_wait_time_min=b, SWW=True)

# LJW
t = 3 # Short job threshold in minutes
sim.run_LJW(max_wait_time_min=sys.maxsize, short_thresh_min=3, cpd=False)

# Compound
b = 15 # maximum wait time in minutes
t = 3 # Short job threshold in minutes
sim.run_LJW(max_wait_time_min=b, short_thresh_min=t, cpd=True)
```

*3.* The simulator’s output is the mean waiting time w, the effective price P, the fraction of jobs that run on on-demand resources r, and the total cost C.

*4.* Simulator needs to be reset to change either the cluster config or the input trace, we provide following functions to do that. Please check the simulator.py for further details. Note that, simulator needs to be reset between the simulation runs, you can do that by using reset_cluster_config function and pass the current cluster config information (i.e., reserve capacity, cpu per reserve machine, and memory per reserve machine).

```python
reset_cluster_config()
reset_input_trace()
```
