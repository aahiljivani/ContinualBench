import numpy as np

   
def step_success(success):
    return np.array(success).mean()

    
def task_performance(success_buffer):
        return np.array(success_buffer).mean()


