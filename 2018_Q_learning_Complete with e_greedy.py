# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:03:19 2018

@author: Jikhan Jeong
"""

## 2018Grid World Q learning by Jikhan Jeong
## 10 x 10 environment
## Discount Factor = 0.9
## Learning Rate   = 0.01
## Action          = R,L,U,D
## Reward   -1, 1(=Goal)

## 1__________________Greedy
## Policy* argmax(Q(s,a))

import numpy as np
# import matplotlib.pyplot as plt


discount_factor = 0.9 # discount factor
learning_rate   = 0.01
num_episodes = 1000

## Environment

grid =np.zeros([10,10])

grid[2,1:5] = -np.inf # Wall  ---> Back to previous state y axis, x axis
grid[2,6:9] = -np.inf # Wall
grid[3:8,4] = -np.inf # Wall 
 
grid[3,3]   = -1  # Penalty
grid[7,3]   = -1  # Penalty
grid[4,5:7] = -1  # Penalty
grid[5,6]   = -1  # Penalty
grid[7,5:7] = -1  # Penalty
grid[5:7,8] = -1  # Penalty

grid[5,5]   =  1  # Goal ---> Back to start grid[0,0], Terminal Point
grid

start = [0,0]                                        # Start Top left

action_move =[[-1,0],[1,0],[0,-1],[0,1]] # up =0, down=1, left=2, right=3

def reset(state):                                    # Goal--->Statring point
    state = start
    return state
    
def step(state, action): ######################################################### why not?
    next_state = [state[0] + action_move[action][0], # x(t) + x(action)
                  state[1] + action_move[action][1]] # y(t) + y(action) 
    if next_state[0] == -1:
       next_state =[state[0],state[1]]                     # up    = 0
    if next_state[0] == 10:
       next_state =[state[0],state[1]]                     # down  = 1
    if next_state[1] == -1:
       next_state =[state[0],state[1]]                     # left  = 2
    if next_state[1] == 10:
       next_state =[state[0],state[1]]                     # right = 3  
    if grid[next_state[0],next_state[1]] == -np.inf:
#       next_state = [state[0], state[1]]
       next_state =[state[0],state[1]]     ## More strong Setting (=Reduction Approach) 
    return next_state

def possible_action(state):                      # Making a bounded set of actions based on its state in girdworld
    possible_actions =[]           
    if (state[0] > 0) and (grid[state[0]-1,state[1]]!=-np.inf):
       possible_actions.append(0)                # up    = 0
    if (state[0] < 9) and (grid[state[0]+1,state[1]]!=-np.inf):
       possible_actions.append(1)                # down  = 1
    if (state[1] > 0) and (grid[state[0],state[1]-1]!=-np.inf):
       possible_actions.append(2)                # left  = 2
    if (state[1] < 9) and (grid[state[0],state[1]+1]!=-np.inf):
       possible_actions.append(3)                # right = 3   
    possible_actions =np.array(possible_actions) #, dtype=int)
    return possible_actions

Q = np.zeros([100,4]) # Start with initial Q-function (e.g. all zero, 100 state 4 possible action)

# reward_list =[]       # Saving the results as a list

def e_greedy(e):
    for i in range(num_episodes): # num_episodes = 100
    # Reset environment and get first new observation
        state = start ## starting point [0,0] Top left
#       rALL=0     
    # Q-table learning algorithm
        if state == [5,5]:                                       # reward -->start
           state = reset(state)        
#    while not ((state ==[5,5]) or (grid[state[0],state[1]]==-np.inf)):   
        while not state == [5,5]:         
        # Choose an action by e greedy
            q_state= state[0]*10 + state[1] # 0-99, 
            if np.random.rand(1) < e:                         # explore p
                action = np.random.choice(possible_action(state)) # random move in state
            else:  
                action = np.argmax(Q[q_state,possible_action(state)])          # exploit 1-p        
           # update Q table with new knowledge using learning rate
            next_state = step(state, action)
            reward     = grid[next_state[0],next_state[1]]
            q_next_state =next_state[0]*10 + next_state[1]
            Q[q_state, action] = Q[q_state, action]  + learning_rate*(reward + discount_factor*np.max(Q[q_next_state,:]) - Q[q_state,action])# : means all actions (R,L,U,D)       
#            rALL +=reward # reward in each episode
            state = next_state          
    return  print("e-greedy, e ={}".format(e)), print(Q)          

e_greedy(0.1)
e_greedy(0.2)
e_greedy(0.3)




for i in range(num_episodes): # num_episodes = 100
    # Reset environment and get first new observation
    state = start ## starting point [0,0] Top left
#       rALL=0     
    # Q-table learning algorithm
    if state == [5,5]:                                       # reward -->start
       state = reset(state)        
#    while not ((state ==[5,5]) or (grid[state[0],state[1]]==-np.inf)):   
    while not state == [5,5]:         
        # Choose an action by e greedy
        q_state= state[0]*10 + state[1] # 0-99, 
        if np.random.rand(1) < 0.1:                         # explore p
            action = np.random.choice(possible_action(state)) # random move in state
        else:  
            action = np.argmax(Q[q_state,possible_action(state)])          # exploit 1-p        
           # update Q table with new knowledge using learning rate
        next_state = step(state, action)
        reward     = grid[next_state[0],next_state[1]]
        q_next_state =next_state[0]*10 + next_state[1]
        Q[q_state, action] = Q[q_state, action]  + learning_rate*(reward + discount_factor*np.max(Q[q_next_state,:]) - Q[q_state,action])# : means all actions (R,L,U,D)       
#            rALL +=reward # reward in each episode
        state = next_state
          
print(Q)  














# reward_list.append(rALL) # put how many time success during episode {1 = success, 0=failure}
# print("Success rate: " + str(sum(reward_list)/num_episodes)) 
# print("Final Q table Values")
# print("Left Down Right Up")

# plt.bar(range(len(reward_list)),reward_list, color="blue")
# plt.show()

## reference : 
## Learning Resource and reference
## Video lecture on Q learning : https://www.youtube.com/watch?v=qPE4CPQY7mc&t=2s
## TD updata each time
## Video Lecture on Gridworld : https://www.youtube.com/watch?v=bHeeaXgqVig&t=16s
## Reinformcement Q learning lecture : https://www.youtube.com/watch?v=Vd-gmo-qO5E&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=4
## Simple Q learning Demo : https://www.youtube.com/watch?v=yOBKtGU6CG0&index=5&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
## Open AI Set up: http://jaynewho.com/post/10        
## https://www.youtube.com/watch?v=MQ-3QScrFSI&index=6&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
## https://github.com/ankonzoid/LearningX/blob/master/classical_RL/gridworld/gridworld.py



