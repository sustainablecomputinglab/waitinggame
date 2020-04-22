from datetime import datetime

import numpy as np
import pandas as pd


def generate_synthetic_data(mu_service: float, lambda_arrival: float,
                            start_time: datetime, no_of_jobs: int, year_: int,
                            save_to_disk:bool = False, save_loc:str = None) -> pd.DataFrame:
   dist_service = np.random.exponential(scale=1/mu_service, size=no_of_jobs)
   dist_arrival = np.random.exponential(scale=1/lambda_arrival, size=no_of_jobs)
   data = pd.DataFrame({'arrival': dist_arrival, 'wallclock_runtime_sec':dist_service})
   data['time'] = data['arrival'].cumsum() 
   data['time'] = start_time + pd.to_timedelta(data['time'], unit='s')
   data['time'] = pd.to_datetime(data['time'], unit='s')

   # add new columns to the data here 
   # userid,wallclock_limit_sec,wallclock_runtime_sec,num_cores,total_MB_req,status,time
   data['userid'] = [0 for _ in range(no_of_jobs)]
   data['wallclock_limit_sec'] = [0 for _ in range(no_of_jobs)]
   data['num_cores'] = [1 for _ in range(no_of_jobs)]
   data['cpu_dominant'] = [1 for _ in range(no_of_jobs)]
   data['total_MB_req'] = [4096 for _ in range(no_of_jobs)]
   data['status'] = ["DONE" for _ in range(no_of_jobs)]

   data = data[['userid','wallclock_runtime_sec', 'wallclock_limit_sec',
                'cpu_dominant','num_cores','total_MB_req','status','time']]

   # Assign the job ids
   data['job_id'] = data.index

   # Save to disk
   if save_to_disk and save_loc is not None:
      data.to_hdf(save_loc + 'synthetic-trace-' + str(year_) + ".h5",
                    key='stage', mode='w', complevel=9, complib='zlib')

   return data

if __name__ == "__main__":
   s = 100
   lambda_s = 0.002
   lambda_a = lambda_s * s
   year_ = 2025
   start_t = datetime(year=2025, month=1, day=1, hour=0, minute=0, second=0)
   N = 2000000
   # call the function
   generate_synthetic_data(lambda_s, lambda_a, start_t, N, year_, save_to_disk=True, save_loc="../synthetic_traces/")
