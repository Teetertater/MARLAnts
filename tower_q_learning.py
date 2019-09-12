import gym
import tower
import numpy as np
import math
import os

env = gym.make('Tower-v0')

state_space = env.observation_space.n
action_space = env.action_space.n
num_agents = env.num_agents

playback = True                           # Playback uses the last q table to run a demo if True
episodes = 10                             # Amount of episodes to train/display
update_q = False if playback else True
fps = 5                                   # Controls speed of render

last_file = 1
while os.path.exists("qtable_%s.npy" % last_file):
    last_file += 1
if playback:    
    qtable = np.load("qtable_{}.npy".format(last_file-1))
else:
    qtable = np.zeros((state_space, action_space))


epsilon = 0.0 if playback else 0.0          #Greed 100%
epsilon_min = 0.005                         #Minimum greed 0.05%
epsilon_decay = 0.99993/50000               #Decay multiplied with epsilon after each episode
max_steps = 100                             #Maximum steps per episode
learning_rate = 0.75
gamma = 0.65


wins, losses = [0]*10, [0]*10
for episode in range(episodes):
 
    # Reset the game state, done and score before every episode/game
    states = env.reset() #Gets current game state
    done = False        #decides whether the game is over
    score = 0

    for _ in range(max_steps):
        #Pygame visualization
        if playback: env.render(mode = "human", fps = fps)

        # With the probability of (1 - epsilon) take the best action in our Q-table
        # Else take a random action
        actions = [np.argmax(qtable[states[i], :]) if np.random.uniform(0, 1) > epsilon else env.action_space.sample() for i in range(num_agents)]
        #if np.random.uniform(0, 1) > epsilon: action = np.argmax(qtable[state, :])
        #else: action = env.action_space.sample()
        
        # Step the game forward, Add up the score
        next_states, reward, done, info = env.step(actions)
        score += reward

        #Calculate win percentages
        if info == "win": 
            wins[math.floor(episode/episodes*10)] += 1
        elif info =="lose":
            losses[math.floor(episode/episodes*10)] += 1
            print("lose")
        if update_q:  # Update our Q-table with our Q-function
            for i in range(num_agents):
                qtable[states[i], actions[i]] = (1 - learning_rate) * qtable[states[i], actions[i]] \
                    + learning_rate * (reward + gamma * np.max(qtable[next_states[i],:]))
 
        # Set the next state as the current state
        states = next_states

        if done:
            break
 
    # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
    if epsilon >= epsilon_min:
        epsilon *= epsilon_decay*episodes
    
    #sys.stdout.write("%d%%\r" % int(episode*100/episodes))
    #sys.stdout.flush()

env.close()
print("win percentages:")
print([round(w/(w+losses[l]), 2) for l,w in enumerate(wins)])

print("q table:")
print(qtable)
if not playback: np.save("qtable_{}".format(last_file), qtable)