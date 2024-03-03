# Model Free Episodic Control

## Description

This is an implementation of Google DeepMind's algorithm *Model Free Episodic Control*, a reinforcement learning algorithm. The algorithm is a gradient-free method to estimate Q-values of an environment by storing all of the observed states and using k-nearest neighbors to search and return a top-k mean of the accumulated reward values. This method is more sample efficient than traditional DQN technique at learning _. For snake, an environment with a relatively small state space, the unprocessed observations are sufficent. For environments with larger state spaces, a dimensionality reduction technique, such as random projection, must be used.

The project has three main components:
- Replay memory is used to store each state transition and calculate discounted rewards for each episode. 
- The Episodic Controller is the brain of the algorithm which stores and sorts the states/reward matrix.
- Train/evaluate for each environment type.

## Getting Started

Clone the repo
```
git clone https://github.com/stephkno/Model-Free-Episodic-Control.git
python -m venv mfec
mfec/bin/pip -r requirements.txt
```

To train
```
mfec/bin/python train_snake.py
```

To evaluate training weights
```
mfec/bin/python evaluate_snake.py
```
