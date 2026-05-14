from random import seed
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from continual_bench.envs import ContinualBenchEnv
import random


# WE WILL MAKE TWO CLASSES ONE THAT CHECKS IF THE CONTINUALBENCHENV IS VALID AND ONE THAT MAKES THE VECTORIZED ENVIRONMENT.
# def my_check_env():
#     from gymnasium.utils.env_checker import check_env
#     env = gym.make('airplane-boarding-v0', render_mode=None)
#     check_env(env.unwrapped)

# if __name__ == "__main__":
#     # my_check_env()

class ContinualBenchVecEnv:
    def __init__(self, num_envs, seed=0, vec_env_cls=SubprocVecEnv, task="sequential"):
        self.env = ContinualBenchEnv(render_mode=None, seed=seed)
        self.task = task
        self.task_list = ["button", "door", "window", "faucet", "peg", "block", "sequential", "random"]
        self.num_envs = num_envs
        self.seed = seed
        self.vec_env_cls = vec_env_cls
        
        self.task_sequence = ["button", "door", "window", "faucet", "peg", "block"]
        
        self.current_task = None
        

    def make_env(self):
        """
        makes parallel task execution for sequential random and 
        single tasks in continual-bench for gymnasium style envs
        
        """
        env = self.env
        comp_envs = dict()
        if self.task in self.task_sequence:
            self.current_task = self.task
            env.set_task(self.task)
            comp_envs[self.current_task] = make_vec_env(env_id = GymV21CompatibilityV0(env), n_envs=self.num_envs, vec_env_cls=self.vec_env_cls)
        
        elif self.task == "sequential":
            for i in range(len(self.task_sequence)):
                self.current_task = self.task_sequence[i]
                set_task = env.set_task(self.current_task)
                comp_envs[self.current_task] = make_vec_env(GymV21CompatibilityV0(set_task), n_envs=self.num_envs, vec_env_cls=self.vec_env_cls)
        elif self.task == "random":
            tasks = self.task_list[:-2]
            random.shuffle(tasks)
            for task in tasks:
                self.current_task = task
                set_task = env.set_task(self.current_task)
                comp_envs[self.current_task] = make_vec_env(GymV21CompatibilityV0(set_task), n_envs=self.num_envs, vec_env_cls=self.vec_env_cls)
        else:
            raise ValueError(f"Task {self.task} not found. Available tasks: {self.task_list}")

        return comp_envs

    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Action shape must be {self.num_envs}, got {action.shape[0]}"
            )
        return self.env.step(action)
        # obs, rewards, dones, infos = self.env.step(action)
        # TODOjust want to return the current task reward 


    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space()




        
        