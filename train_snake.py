#from nes_py.wrappers import JoypadSpace
#import gym_tetris
#from gym_tetris.actions import MOVEMENT
import gym
import gym_snake_game
import numpy

import EpisodicController
import ReplayMemory

options = {
    'fps': 60,
    'max_step': 500,
    'init_length': 4,
    'food_reward': 2.0,
    'dist_reward': None,
    'living_bonus': 0.0,
    'death_penalty': -1.0,
    'width': 40,
    'height': 40,
    'block_size': 20,
    'background_color': (0, 0, 0),
    'food_color': (255, 90, 90),
    'head_color': (197, 90, 255),
    'body_color': (89, 172, 255),
}

env_name = "Snake-v0"
env = gym_snake_game.make(env_name, **options)

replay = ReplayMemory()
controller = EpisodicController(env_name, env.observation_space, env.action_space)

epoch = 0

max_return = -99999
controller.load()
controller.epsilon = 0.005

def preprocess(state):
    return numpy.array(state)

while True:
    epoch += 1

    replay.reset()
    state, info = env.reset()
    done = False

    total_return = 0

    while not done:
        state = preprocess(state)

        action = controller.process_observation(state)

        observation = env.step(action)

        controller.frames_run += 1
        new_state, reward, done, info, _ = observation
        total_return += reward

        replay.append(state, action, reward, done)
        state = new_state
    
    if total_return > max_return:
        max_return = total_return
    controller.save()

    print("Game: {} Frames: {} Final Score: {} High score: {} Epsilon:{}".format(epoch, controller.frames_run, total_return, max_return, controller.epsilon))
    controller.update_policy(replay)
