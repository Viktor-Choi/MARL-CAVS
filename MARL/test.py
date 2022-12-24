#%% 
import sys
sys.path.insert(0,"../highway-env")
import highway_env
import gym
import importlib
from matplotlib import pyplot as plt
from common.utils import VideoRecorder

#%%
# 重载没有用，不懂为什么
importlib.reload(highway_env)
importlib.reload(highway_env.envs)
importlib.reload(gym.envs.registration)
importlib.reload(highway_env.envs.merge_env_v1)  

#%% watching
env = gym.make('merge-v1')
env.reset()

while not env.done:
    action = env.action_type.actions_indexes["IDLE"]
    env.step(action)
    env.render()

env.close()

#%% video
env = gym.make('merge-v1')
env.reset()
rendered_frame = env.render(mode="rgb_array")

video_recorder = VideoRecorder('test.mp4',frame_size=rendered_frame.shape,fps=15)
video_recorder.add_frame(rendered_frame)
terminal = False

while not terminal:
    action = env.action_type.actions_indexes["IDLE"]
    _,_,terminal,_ = env.step(action)
    rendered_frame = env.render(mode="rgb_array")
    video_recorder.add_frame(rendered_frame)

assert env.viewer is not None
video_recorder.release()
env.close()
