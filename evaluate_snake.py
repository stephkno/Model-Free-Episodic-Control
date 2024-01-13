#from nes_py.wrappers import JoypadSpace
#import gym_tetris
#from gym_tetris.actions import MOVEMENT

import gym
import numpy
import gym_snake_game

import EpisodicController
import ReplayMemory

options = {
    'fps': 25,
    'max_step': 500,
    'init_length': 4,
    'food_reward': 2.0,
    'dist_reward': None,
    'living_bonus': 0.0,
    'death_penalty': -1.0,
    'width': 40,
    'height': 40,
    'block_size': 20,
    'food_color': (0, 90, 90),
    'head_color': (197, 90, 255),
    'body_color': (89, 172, 255),
}

env_name = "Snake-v0"

env = gym_snake_game.make(env_name, render_mode="human", **options)
controller = EpisodicController(env_name, env.observation_space, env.action_space)
epoch = 0

max_return = 0
controller.epsilon = 0

def preprocess(state):
    return numpy.array(state)

while True:
    
    controller.load()
    epoch += 1

    state, info = env.reset()
    done = False

    step = 0
    total_return = 0
        
    while not done:

        state = preprocess(state)

        action = controller.process_observation(state)
        observation = env.step(action)
        
        new_state, reward, done, info, _ = observation
        total_return += reward

        state = new_state

    if total_return > max_return:
        max_return = total_return

    print("Game: {} Final Score: {} High score: {} Epsilon:{}".format(epoch, total_return, max_return, controller.epsilon))
