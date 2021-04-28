[//]: # (Image References)

[image1]: https://github.com/ga32riv/Project-3-collaboration_and_competition/blob/main/Average%20plot.png "Average Reward"

[image2]: https://github.com/ga32riv/Project-3-collaboration_and_competition/blob/main/Score%20plot.png "Reward each Episode"

# Report

## Goal

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Steps
1. Examine the State and Action Spaces.
2. Take Random Actions in the Environment (to learn how to use the Python API to control the agent and receive feedback from the environment).
3. Implement learning algorithm and check performance.


## Algorithm
This project implements a  method called Multi-Agent Deep Deterministic Policy Gradient (MADDPG). Based on *DDPG* described in the paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) but adapted for a training with multiple agents.
The agents was trained using [a single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a template.

## NN Model Architecture

Actor and Critic are using Adam optimizer

# *Actor* :

- Fully connected layer - input (state size): 2*24 - output: 256 (ReLu activation function)

- Fully connected layer - input: 256 - output: 128 (ReLu activation function)

- Fully connected layer - input: 128 - output: 2 (action size, with tanh activation)

# *Critic* :
- Fully connected layer - input (state size): 2*24 - output: 256 (ReLu activation function)

- Fully connected layer - input: 256 + (2*2) - output: 128 (ReLu activation function)

- Fully connected layer - input: 128 - output: 1

## Parameters
'''
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 1           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 7e-2              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.12         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.5         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 250        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
'''

## Results
```
Goal reached in 1242 episodes!                 
Moving Average: 0.505 over past 100 episodes
Episodes 1350 (52 sec)	Max Reward: 4.700	Moving Average: 0.562
```
The Environment is solved in 1242 episodes!	with a Moving Average of 0.505 over last 100 episodes

![Average Reward][image1]
<img src="https://github.com/ga32riv/Project-3-collaboration_and_competition/blob/main/Score%20plot.png" width="100" height="100">


## Ideas for Future Work
1. Compare the results with other algorithms, for example PPO.
2. Perform a hyperparameter search.
3. Use Priorizited Experience Replay
