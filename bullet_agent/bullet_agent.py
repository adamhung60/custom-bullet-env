import bullet_env
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn as nn

def train(ts):
    env = gym.make("Bullet", render_mode="rgb_array")
    vec_env = DummyVecEnv([lambda: env])
    vec_env = Monitor(vec_env)
    vec_env = VecNormalize(vec_env, 
                              training=True, 
                              norm_obs=True, 
                              norm_reward=True, 
                              #clip_obs=10.0, 
                              #clip_reward=10.0, 
                              #gamma=0.99, 
                              #epsilon=1e-08, 
                              #norm_obs_keys=None
                              )
    
    eval_callback = EvalCallback(vec_env,
                             log_path="logs/", eval_freq=500,
                             deterministic=True, render=False)
    

    model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate = 0.0002)

    model.learn(total_timesteps=ts,progress_bar=True, callback=eval_callback)
    model.save("logs/ppo_bullet")
    vec_env.save("logs/vec_normalize.pkl")


def test(eps):

    vec_env = DummyVecEnv([lambda: gym.make("Bullet", render_mode = "human")])
    
    vec_env = VecNormalize.load("logs/vec_normalize.pkl", vec_env)
    model = PPO.load("logs/ppo_bullet", env=vec_env)

    mean_reward,_ = evaluate_policy(model, vec_env, n_eval_episodes = eps, deterministic = True)
    print("mean reward: ", mean_reward)
        
if __name__ == '__main__':
    #train(2e5)
    test(5)
