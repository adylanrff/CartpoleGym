import gym 
import math
import time
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

MAX_EPISODE = 2000
T_GOAL = 199
STREAK_GOAL = 120

episodes_data = []

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
EXPLORE_DECAY = 10
LEARNING_DECAY = 15

DISCOUNT_FACTOR = 0.99

NUM_BUCKETS = [6,3]
NUM_ACTION = env.action_space.n
NUM_TABLE = NUM_BUCKETS + [NUM_ACTION]

q_table = np.zeros(NUM_TABLE)

def select_action(state, explore_rate):
	if (np.random.random() < explore_rate):
		action = env.action_space.sample()
	else:
		action = np.argmax(q_table[state])

	return action 


def normalize(number,n_min,n_max):
	""" 
		Normalize the data from 0 to 1
	"""
	return ((number-n_min)/(n_max-n_min))

def state_to_bucket(state):
	x_state = state[0]
	xdot_state = state[1]
	theta_state = state[2]
	thetadot_state = state[3]

	# Numbers are based on observation, please observe the problem to know the numbers

	theta_bucket = int(round((NUM_BUCKETS[0]-1)* normalize(theta_state, -0.4, 0.4)))
	thetadot_bucket = int(round((NUM_BUCKETS[1]-1)* normalize(thetadot_state, -0.8, 0.8)))

	if (theta_bucket<=0):
		theta_bucket = 0
	elif (theta_bucket>=NUM_BUCKETS[0]):
		theta_bucket = NUM_BUCKETS[0]-1

	if (thetadot_bucket<0):
		thetadot_bucket = 0
	elif (thetadot_bucket>=NUM_BUCKETS[1]):
		thetadot_bucket = NUM_BUCKETS[1]-1

	# return (x_bucket,theta_bucket, thetadot_bucket)
	return (theta_bucket, thetadot_bucket)


# IMPORTANT
# Decaying learning and explore rate
def get_learning_rate(t):
	return max(MIN_LEARNING_RATE, min(0.5, 1.0 - np.log10((t+1)/LEARNING_DECAY)))
def get_explore_rate(t):
	return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - np.log10((t+1)/EXPLORE_DECAY)))


def train():

	episode = 0
	explore_rate = get_explore_rate(0)
	learning_rate = get_learning_rate(0)
	n_goal = 0
	n_streak = 0
	max_streak = 0

	while episode < MAX_EPISODE:
		observation = env.reset()
		initial_state = state_to_bucket(observation)
	
		for t in range(500):
			env.render()
			time.sleep(0.005)

			action = select_action(initial_state,explore_rate)

			observation, reward, done, info = env.step(action)
			
			state = state_to_bucket(observation)

			max_q = np.amax(q_table[state])
			q_table[initial_state + (action,)] += learning_rate*(reward + DISCOUNT_FACTOR*(max_q) - q_table[initial_state + (action,)]) 

			initial_state = state

			if done:
				print(q_table)
				print("Episode {} finished after {} timesteps".format(episode,t+1))
				print("Learning rate : {}".format(learning_rate))
				print("Explore rate : {}".format(explore_rate))
				episodes_data.append(t)
				if (t >= T_GOAL):
					n_goal+=1
					print("Passed goal!")
					n_streak+=1
					if (max_streak < n_streak):
						max_streak = n_streak
				else:
					n_streak = 0

				print("Solved : {}".format(n_goal))
				print("Max Streaks : {}".format(max_streak))
				print("Streaks : {}".format(n_streak))		
				break

		if (n_streak >= STREAK_GOAL):
			print("Max Streaks : {}".format(max_streak))
			print("Streaks : {}".format(n_streak))
			break

		episode+=1
		learning_rate = get_learning_rate(episode)
		explore_rate = get_explore_rate(episode)

if __name__ == '__main__':
	train()
	plt.plot(episodes_data)
	plt.show()
