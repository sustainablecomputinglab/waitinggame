# Synthetic Traces

Synthetic traces can be generated using the included script in synthetic_trace_generator folder. Synthetic traces included here can be used to reproduce the validation experiments. Note that the traces are in hierarchical data format (hdf).

## Important Data Fields

* *userid*: The user identification number. (Integer)

* *wallclock_runtime_sec*: Actual job runtime in seconds. (Integer)

* *wallclock_limit_sec*: Maximum runtime limit specified by the user. (Integer)

* *num_cores*: Number of CPU requested for the job. (Integer)

* *total_MB_req*: Memory in MB requested for the job. (Integer)

* *status*: Status of the job (String)

* *time*: Timestamp when job queued. (Timestamp)
