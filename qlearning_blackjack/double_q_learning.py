import numpy as np
import random
from IPython.display import clear_output
import gym
import matplotlib.pyplot as plt
# https://rubikscode.net/2020/01/20/double-q-learning-python/

# Globals
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.05

def evaluate(enviroment, q_tables, count_cards=False):
    wins = 0
    for episode in range(0, 100):
        # Reset the enviroment
        state,info = enviroment.reset()
        if not count_cards:
            state = (state[0], state[1], state[2])        

        # Initialize variables
        terminated = False

        reward = 0
        while not terminated:
            # Pick action a....
            q_table = np.sum([t[state] for t in q_tables])
            max_q = np.where(q_table == np.max(q_table))[0]
            action = np.random.choice(max_q)

            # ...and get r and s'
            next_state, reward, terminated, _,_ = enviroment.step(action)

            state = next_state
            if not count_cards:
                state = (state[0], state[1], state[2])

        if reward > 0:
            wins += 1

    return wins / 100

def q_learning(enviroment, num_of_episodes=1000, count_cards=False):
    # Initialize Q-table
    if count_cards:
        q_table = np.zeros([32,11,2,11,5,2])
    else:
        q_table = np.zeros([32,11,2,2])
    rewards = np.zeros(num_of_episodes)
    win_rate = np.zeros(num_of_episodes // 100)

    wins = 0
    for episode in range(0, num_of_episodes):
        # Reset the enviroment
        state,info = enviroment.reset()

        if not count_cards:
            state = (state[0], state[1], state[2])

        # Initialize variables
        terminated = False
        
        states  = []
        actions = []
        reward  = 0
        while not terminated:
            states.append(state)

            # Pick action a....
            if np.random.rand() < EPSILON:
                action = enviroment.action_space.sample()
            else:
                # print(state)
                # print(q_table[state])
                max_q = np.where(q_table[state] == np.max(q_table[state]))[0]
                # print(max_q)
                action = np.random.choice(max_q)

            actions.append(action)

            # ...and get r and s'
            next_state, reward, terminated, _,_ = enviroment.step(action)
            # print(reward)

            state = next_state
            if not count_cards:
                state = (state[0], state[1], state[2])

        states.append(state)
           
        # Update Q-Table
        for i in range(len(states) - 1):
            state = states[i]
            action = actions[i]
            
            reward_i = q_table[state][action] + ALPHA*(reward - q_table[state][action])
            q_table[state][action] = round(reward_i, 3)

        rewards[episode] += reward

        if reward > 0:
            wins += 1
        
        if episode % 100 == 0:
            wr = evaluate(enviroment, [q_table])
            print("Episode: ", episode)
            print("Win rate: ", evaluate(enviroment, [q_table]))
            print("")
            win_rate[episode // 100] = wr


    return rewards, win_rate, q_table


def double_q_learning(enviroment, num_of_episodes=1000, count_cards=False):
    if count_cards:
        q_a_table = np.zeros([32,11,2,11,5,2])
        q_b_table = np.zeros([32,11,2,11,5,2])
    else:
        q_a_table = np.zeros([32,11,2,2])
        q_b_table = np.zeros([32,11,2,2])

    rewards = np.zeros(num_of_episodes)
    win_rate = np.zeros(num_of_episodes // 100)
    for episode in range(0, num_of_episodes):
        # Reset the enviroment
        state,info = enviroment.reset()

        if not count_cards:
            state = (state[0], state[1], state[2])

        # Initialize variables
        terminated = False

        states  = []
        actions = []
        reward  = 0
        while not terminated:
            states.append(state)

            # Pick action a....
            if np.random.rand() < EPSILON:
                action = enviroment.action_space.sample()
            else:
                q_table = q_a_table[state] + q_b_table[state]
                max_q = np.where(q_table == np.max(q_table))[0]
                action = np.random.choice(max_q)

            actions.append(action)

            # ...and get r and s'
            next_state, reward, terminated, _,_ = enviroment.step(action)

            state = next_state
            if not count_cards:
                state = (state[0], state[1], state[2])

        states.append(state)

        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i+1]
            action = actions[i]        
            # Update(A) or Update (B)
            if np.random.rand() < 0.5:
                # If Update(A)
                q_a_table[state][action] += ALPHA * (reward - q_a_table[state][action])
                    
            else:
                # If Update(B)
                q_b_table[state][action] += ALPHA * (reward - q_b_table[state][action])

        rewards[episode] += reward
        
        if episode % 100 == 0:
            wr = evaluate(enviroment, [q_a_table, q_b_table])
            print("Episode: ", episode)
            print("Win rate: ", evaluate(enviroment, [q_a_table, q_b_table]))
            print("")
            win_rate[episode // 100] = wr
                
    return rewards, win_rate, q_a_table, q_b_table

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")

    # q_reward, q_table = q_learning(env)
    # print(q_table)
    dq_reward, q_a_table, q_b_table = double_q_learning(env)
    plt.plot(dq_reward)
    plt.xlabel("Rounds")
    plt.ylabel("Rewards")
    plt.title("double Q-learning rewards")
    plt.savefig("double q_learning")
    # plt.show()