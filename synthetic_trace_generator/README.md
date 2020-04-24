# Synthetic Trace Generator

Synthetic trace generator generates the traces that can be used as input to the simulator. The trace generator takes inputs as job arrival rate, job service rate, number of jobs, and optionally save location (to save the trace on disk). In the generated job trace, arrivals form a single queue and are governed by a Poisson process, and job service times are exponentially distributed (M/M/c queue). In the current implementation, we save the trace in hdf format ([https://en.wikipedia.org/wiki/Hierarchical_Data_Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)). Each job entry includes its submission time, user ID, requested number of cores and memory, and running time. In the synthetic trace, we assign a static value for both requested number of cores and memory.

## Usage

1. synthetic_trace.py implements the synthetic trace generator. Run the synthetic_trace.py (check main function) with respective parameter value to generate synthetic trace.
2. Example: (Same as main function in synthetic_trace.py)

```python
s = 100
lambda_s = 0.002
lambda_a = lambda_s * s
year_ = 2025
start_t = datetime(year=2025, month=1, day=1, hour=0, minute=0, second=0)
N = 2000000
# call the function
generate_synthetic_data(lambda_s, lambda_a, start_t, N, year_, save_to_disk=True, save_loc="../synthetic_traces/")
```
