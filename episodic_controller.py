import numpy
import random

#
# https://arxiv.org/pdf/1606.04460.pdf

class EpisodicController:
    def __init__(self, env_name, observation_space, action_space):
        
        # initialize episodic controller matrices
        self.action_space = action_space
        self.observation_space = observation_space
        
        # initialize internal parameter space
        self.action_size = self.action_space.n
        self.observation_size = observation_space.shape[0]

        # initialize hyperparameters
        # k nearest neighbors
        self.k = 5
        self.name = env_name

        # maximum states in controller
        self.n_max = 1000000
        self.n_states = 0
        self.frames_run = 0

        # epsilon greedy
        self.epsilon = 0.005
        self.epsilon_decay = 1

        self.state_matrix = numpy.zeros([1,self.observation_size])
        self.action_matrix = numpy.zeros([1,self.action_size])
        

    def save(self):
        filename = self.name + '.npz'
        numpy.savez(filename, self.state_matrix, self.action_matrix, self.frames_run)

    def load(self):
        filename = self.name + '.npz'
        try:
            npzfile = numpy.load(filename)
            self.state_matrix = npzfile['arr_0']
            self.action_matrix = npzfile['arr_1']
            self.frames_run = npzfile['arr_2']

        except:
            print("File not found - initializing")

    def get_num_states(self):
        pass

    # (train) update episodic controller with replay
    def update_policy(self, replay):
        replay.discount_rewards()
        self.epsilon *= self.epsilon_decay
        
        for step in reversed(replay.memory):
            state, action, reward, _ = step

            # create reward vector
            reward_vector = numpy.zeros(self.action_space.n)
            reward_vector[action] = reward

            # insert new state if (st, at) !∈ QEC
            # determine if state is in state matrix
            index = numpy.flatnonzero((state==self.state_matrix).all(1))
            
            # add new vector to matrices if state is novel
            if index.shape[0] == 0:
                # add new state and reward vector to internal matrices
                self.state_matrix = numpy.concatenate((self.state_matrix, numpy.expand_dims(state,axis=0)),axis=0)
                self.action_matrix = numpy.concatenate((self.action_matrix, numpy.expand_dims(reward_vector,axis=0)),axis=0)
                index = numpy.flatnonzero((state==self.state_matrix).all(1))
            
            # update reward value of state at action index
            self.action_matrix[index][0][action] = max(self.action_matrix[index][0][action], reward)

        #self.state_matrix = numpy.resize(self.state_matrix, (self.n_max, self.observation_space.shape[0]))
        #self.action_matrix = numpy.resize(self.action_matrix, (self.n_max, self.action_space.n))

    # process observation and return discrete action
    def process_observation(self, state):

        # broadcast state vector to fit state and action matrix
        #print(self.state_matrix)
        #print(self.action_matrix)
        self.n_states = self.state_matrix.shape[0]
        
        # if state in episodic controller
        index = numpy.flatnonzero((state==self.state_matrix).all(1))
        
        if(index.shape[0]>0):
            actions = self.action_matrix[index[0]]
            return numpy.argmax(self.action_matrix[index[0]])

        # find difference matrix for nearest neighbors
        state_matrix = numpy.broadcast_to(state, (self.n_states, self.observation_size))
        difference_matrix = numpy.abs(state_matrix - self.state_matrix)
        
        # find sorted matrix indices
        inds = numpy.expand_dims(numpy.argsort(numpy.sum(difference_matrix,axis=1)),axis=1)
        
        # sort matrix order by taking indices along axis
        self.state_matrix = numpy.take_along_axis(self.state_matrix, inds, axis=0)
        self.action_matrix = numpy.take_along_axis(self.action_matrix, inds, axis=0)

        # find nearest neighbors
        nearest_neighbors = numpy.copy(self.action_matrix)

        # take top k nearest neighbors
        nearest_neighbors = numpy.resize(nearest_neighbors, (self.k, self.action_size))
        actions = numpy.mean(nearest_neighbors, axis=0)

        # epsilon greedy sampling
        sample = random.random()
        if(sample < self.epsilon):
            return self.action_space.sample()
        else:
            return numpy.argmax(actions)
       


