
class ReplayMemory:
    def __init__(self):
        self.reset()
        self.discount_rate = 1
        
    # 
    def reset(self):
        self.memory = []

    # calculate disconted rewards for current replay
    def discount_rewards(self):
        running_sum = 0

        for i,state in enumerate(reversed(self.memory)):
            running_sum = state[2] + (running_sum * self.discount_rate)
            state[2] = running_sum
    
    # add observation to current replay
    def append(self, state, action, reward, done):
        self.memory.append([state, action, reward, done])
