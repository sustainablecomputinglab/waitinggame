"""
Copyright (c) 2020 <Sustainable Computing Lab>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from collections import namedtuple
from typing import Dict, List, Set

"""
named tuple for a job
"""
JOB = namedtuple("JOB", ['job_id', 'cpu', 'mem_gb'])

"""
Class definition for the Node/VM in the cluster

Instance variables:
_type: str; Type of VM/node e.g., "m5.16xlarge"
_nodeID: int; Node ID of the respective node
CPU: int; Total number of CPU cores in the node
MEM_GB: float; Total memory in GB available in the node
current_jobs: Dictionary; key = int (job_id), value = JOB (namedtuple)
idle_cpu: int; current unused CPU capacity
idle_mem_gb: float; current unused memory capacity
"""
class Node:
  def __init__(self, _type: str, _nodeID: int, CPU: int, MEM_GB: float):
    # Node Info
    self._type = _type
    self._nodeID = _nodeID

    # Constants
    self.CPU = CPU
    self.MEM_GB = MEM_GB

    # Variables
    self.current_jobs: Dict[int, JOB] = {}
    self.idle_cpu = CPU
    self.idle_mem_gb = MEM_GB
    self.util_cpu_time = 0

  """
  Checks if the given job can be scheduled on the respective machine.
  Returns True if the job can be scheduled, else False
  """
  def can_schedule(self, cpu_req: int, mem_req_gb: float) -> bool:
    return (cpu_req <= self.idle_cpu and mem_req_gb <= self.idle_mem_gb)

  """
  Schedule the job on the machine.
  Return True if succedded, else False
  """
  def schedule_job(self, cpu_req: int, mem_req_gb: float, job_id: int) -> bool:
    # Check if there are enough idle resources
    if not self.can_schedule(cpu_req, mem_req_gb):
      return False
    
    # 1. Update the idle resource values
    self.idle_cpu -= cpu_req
    self.idle_mem_gb -= mem_req_gb
    
    # 2. Add the job_id to current_jobs
    self.current_jobs[job_id] = JOB(job_id, cpu_req, mem_req_gb)
    return True

  """
  Remove the finished job on the respective node and add back the compute capacity.
  Return True if the job is running in the respective node, else False
  """
  def remove_job(self, job_id: int) -> bool:
    # Check if the job is running in the respective node
    if job_id not in self.current_jobs:
      return False
    
    # Get the job
    job = self.current_jobs[job_id]

    # Update the idle resource values
    self.idle_cpu += job.cpu
    self.idle_mem_gb += job.mem_gb

    # Delete the key
    del self.current_jobs[job_id]
    return True

  """
  Track the total used cpu time in seconds; 
  Used to compute the utilization of a node
  """
  def update_cpu_time_utilized(self, cpu: int, runtime_sec: int) -> None:
    self.util_cpu_time += cpu * runtime_sec
