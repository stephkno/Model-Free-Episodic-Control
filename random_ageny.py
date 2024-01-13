#from nes_py.wrappers import JoypadSpace
#import gym_tetris
#from gym_tetris.actions import MOVEMENT
import gym
import gym_snake_game

# classic control
env = gym.make("Snake-v0", render_mode="human")

epoch = 0
max_return = -99999

while True:
    epoch += 1
    print("Iteration {}".format(epoch))

    state = env.reset()
    done = False

    step = 0
    total_return = 0
        
    while not done:
        action = env.action_space.sample()

        observation = env.step(action)
        print(observation)
        
        new_state, reward, done, _, _ = observation

        total_return += reward

        state = new_state

    if total_return > max_return:
        max_return = total_return

