# built off 18_rl file then uses below
# https://www.youtube.com/watch?v=bD6V3rcr_54&list=PLgNJO2hghbmjlE6cuKMws2ejC54BTAaWV&index=3
# https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment/blob/main/OpenAI%20Custom%20Environment%20Reinforcement%20Learning.ipynb

import pandas as pd
import numpy as np
from numpy import random
import tensorflow as tf
from gym import Env
from gym.spaces import Discrete, Box
from collections import deque
import matplotlib.pyplot as plt
from scipy import stats


# set seed and some global constants
seed = 42
np.random.seed(seed)
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error
# run, pass, kick
n_outputs = 3  # == env.action_space.n
# yds_to_goal, down, yds_to_first
n_inputs = 3

# ydsToGo_ep_intercept = 5.211187389603027
# ydsToGo_coef = -0.06155063

# the function provides the other team's expected points if there is a change of possession.
def calc_ep(ydsToGo, ydsToGo_ep_intercept=5.211187389603027, ydsToGo_coef=-0.06155063):
    return ydsToGo_ep_intercept + ydsToGo * ydsToGo_coef

# this is the other team's expected points after a score. It assumes they have a touchback and no return
touchback_ep = calc_ep(80)


class FootballEnv(Env):
    init_yds_to_goal = 80
    init_down = 0
    init_yds_to_first = 10

    # https://www.pro-football-reference.com/years/NFL/passing.htm
    # completion percent, sack rate above
    # https: // www.pro - football - reference.com / years / NFL / index.htm
    # fmb_rate = fl / (cmp + rush_att) <- should consider sacks because you are giving chance of fumble when sacked
    def __init__(self, comp_percent=.642, int_rate=.023,
                 sack_rate=.067, sack_avg=-5, sack_sd=1
                 , fmb_rate=.5 / (21.4 + 27.3)):
        self.action_space = Discrete(n_outputs)
        self.observation_space = Box(low=0, high=100, shape=(n_inputs,))
        self.state = [FootballEnv.init_yds_to_goal, FootballEnv.init_down, FootballEnv.init_yds_to_first]
        self.yds_to_goal = FootballEnv.init_yds_to_goal
        self.down = FootballEnv.init_down
        self.yds_to_first = FootballEnv.init_yds_to_first
        self.comp_percent = comp_percent
        self.int_rate = int_rate
        self.sack_rate = sack_rate
        self.sack_avg = sack_avg
        self.sack_sd = sack_sd
        self.fmb_rate = fmb_rate

    def step(self, action):

        self.down += 1
        done = False
        score = 0
        reward = 0
        yds = 0
        turnover = False

        # kick
        if action == 2:
            done = True
            prob = random.rand()
            # made fg
            # https://www.reddit.com/r/nfl/comments/d4h2r0/kicker_accuracy_accounting_for_distance/
            if ((self.yds_to_goal < 60) & (self.yds_to_goal >= 50) & (prob <= .54)) \
                    | ((self.yds_to_goal < 50) & (self.yds_to_goal >= 40) & (prob <= .79)) \
                    | ((self.yds_to_goal < 40) & (self.yds_to_goal >= 30) & (prob <= .90)) \
                    | ((self.yds_to_goal < 30) & (prob <= .95)):
                yds = self.yds_to_goal
                self.yds_to_goal = 0
                self.yds_to_first = 0
                score = 3
                reward = score - touchback_ep
            # missed fg
            else:
                score = 0
                reward = score - calc_ep(100 - self.yds_to_goal)
        # run
        elif action == 0:
            yds = stats.lognorm.rvs(s=0.25558500132474415, loc=-15.189614313924665, scale=18.92331909863129, size=1)[0]
            # fumble?
            if np.random.rand() <= self.fmb_rate:
                turnover = True
        # pass
        elif action == 1:
            # sack?
            if np.random.rand() <= self.sack_rate:
                yds = random.normal(loc=self.sack_avg, scale=self.sack_sd)
                # fumble?
                if np.random.rand() <= self.fmb_rate:
                    turnover = True
            else:
                if self.yds_to_goal <= 20:
                    air_yards = \
                        stats.lognorm.rvs(s=0.18033921317690882, loc=-20.781271075327744, scale=27.083791999969066,
                                          size=1)[
                            0]
                else:
                    air_yards = \
                        stats.lognorm.rvs(s=0.3868256330626365, loc=-11.72666357691477, scale=21.714746429166233,
                                          size=1)[0]
                prob = np.random.rand()
                # completion?
                if prob <= self.comp_percent:
                    yds = air_yards
                    # fumble?
                    if np.random.rand() <= self.fmb_rate:
                        yds = air_yards
                        turnover = True
                # int?
                elif prob >= (1 - self.int_rate):
                    yds = air_yards
                    turnover = True
                # incomplete
                else:
                    yds = 0
        # if action is not kick
        if action != 2:
            self.yds_to_goal = max(self.yds_to_goal - yds, 0)
            self.yds_to_first -= yds
            # first down?
            if self.yds_to_first <= 0:
                self.down = FootballEnv.init_down
                self.yds_to_first = min(self.yds_to_goal, self.init_yds_to_first)
            # turnover on downs?
            elif self.down >= (FootballEnv.init_down + 4):
                turnover = True
            # td?
            if self.yds_to_goal == 0:
                score = 7
                reward = score - touchback_ep
                done = True
            if turnover == True:
                score = 0
                # should take touchback into account
                reward = score - calc_ep(100 - self.yds_to_goal)
                done = True

        # Set placeholder for info
        info = {}

        self.state = [self.yds_to_goal, self.down, self.yds_to_first]

        # Return step information
        # return self.state, reward, done, info
        return self.state, reward, done, info, yds, score

    def render(self):
        # Implement viz
        pass

    # def reset(self):
    def reset(self, yds_to_goal=init_yds_to_goal):
        self.yds_to_goal = yds_to_goal
        self.down = self.init_down
        self.yds_to_first = self.init_yds_to_first
        self.state = [self.yds_to_goal, self.init_down, self.init_yds_to_first]
        return self.state


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # random action
    else:
        # Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        Q_values = model.predict([state], verbose=0)[0]
        return Q_values.argmax()  # optimal action according to the DQN


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    # return [
    #     np.array([experience[field_index] for experience in batch])
    #     for field_index in range(6)
    # ]  # [states, actions, rewards, next_states, dones, truncateds]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]  # [states, actions, rewards, next_states, dones]


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    # next_state, reward, done, truncated, info = env.step(action)
    next_state, reward, done, info, yds, score = env.step(action)
    # replay_buffer.append((state, action, reward, next_state, done, truncated))
    replay_buffer.append((state, action, reward, next_state, done))
    # return next_state, reward, done, truncated, info
    return next_state, reward, done, info, action, yds


def training_step(batch_size=32, discount_factor=0.95):
    experiences = sample_experiences(batch_size)
    # states, actions, rewards, next_states, dones, truncateds = experiences
    states, actions, rewards, next_states, dones = experiences
    if rl_type == 'dqn':
        next_Q_values = model.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
    elif rl_type == 'fixed_dqn':
        next_Q_values = target.predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
    elif rl_type == 'double_dqn':
        next_Q_values = model.predict(next_states, verbose=0)  # ≠ target.predict()
        best_next_actions = next_Q_values.argmax(axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
        max_next_Q_values = (target.predict(next_states, verbose=0) * next_mask).sum(axis=1)
    # runs = 1.0 - (dones | truncateds)  # episode is not done or truncated
    runs = 1.0 - (dones)
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Define a simple sequential model
# https://www.tensorflow.org/tutorials/keras/save_and_load
def create_model():
    input_shape = [n_inputs]  # == env.observation_space.shape

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
        tf.keras.layers.Dense(32, activation="elu"),
        tf.keras.layers.Dense(n_outputs)
    ])

    return model


def train(episodes=500, steps=200, batch_size=32, discount_factor=.95, target_update_freq=50, ep_ticker='y',
          ep_chart='y'):
    rewards = []
    best_score = 0
    for episode in range(episodes):
        # obs, info = env.reset()
        obs = env.reset()
        for step in range(steps):
            # epsilon = max(1 - episode / 500, 0.01)
            epsilon = max(1 - episode / (episodes - 100), 0.01)
            # obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
            obs, reward, done, info, action, yds = play_one_step(env, obs, epsilon)
            # if done or truncated:
            #     break
            if done:
                break

        # # extra code – displays debug info, stores data for the next figure, and
        # #              keeps track of the best model weights so far
        # print(f"\rEpisode: {episode + 1}, Steps: {step + 1}, eps: {epsilon:.3f}",
        #       end="")
        if ep_ticker == 'y':
            print(f"\rEpisode: {episode + 1}, Reward: {reward}, Steps: {step + 1}, eps: {epsilon:.3f}", end="")
        rewards.append(reward)
        if reward >= best_score:
            best_weights = model.get_weights()
            best_score = reward
        # example 50 episodes was ~1.1K but for this 50 episodes is ~150, should really be times 8 but going with 4
        # if episode > (50):
        if episode > (200):
            # training_step()
            training_step(batch_size=batch_size, discount_factor=discount_factor)
            if (rl_type == 'fixed_dqn') | (rl_type == 'double_dqn'):
                # if episode % 50 == 0:
                if episode % target_update_freq == 0:
                    target.set_weights(model.get_weights())

    if ep_chart == 'y':
        # extra code – this cell generates and saves Figure 18–10
        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        plt.grid(True)
        # save_fig("dqn_rewards_plot")
        plt.show()

    return best_weights, rewards


# train and test have too much redundancy
def test(episodes=50, steps=200, show_states='y'):
    reward_tot = 0
    for episode in range(episodes):
        # obs, info = env.reset()
        obs = env.reset()
        for step in range(steps):
            if show_states == 'y':
                print('down:', obs[1] + 1)
                print('yds to 1st', obs[2])
                print('yds to goal', obs[0])
            epsilon = 0.01
            # obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
            obs, reward, done, info, action, yds = play_one_step(env, obs, epsilon)
            if show_states == 'y':
                print('action', action)
                print('yds', yds)
            # if done or truncated:
            #     break
            if done:
                reward_tot += reward
                if show_states == 'y':
                    print('reward:{reward}'.format(reward=reward))
                break
    avg_reward = reward_tot / episodes
    print()
    print('avg reward: {avg_reward}'.format(avg_reward=avg_reward))
    print()
    return avg_reward


def simple_strat(state):

    run_threshold = 3.5

    down = state[1] + 1
    yds_to_first = state[2]
    yds_to_goal = state[0]
    # kick if fourth down
    if (down == 4) & (yds_to_goal < 60):
        action = 2
    elif down >= 3:
        if yds_to_first <= run_threshold:
            action = 0
        else:
            action = 1
    else:
        if np.random.rand() <= .5:
            action = 1
        else:
            action = 0
    return action


def play(possessions=4, team2_strat='simp', show_states='y'):
    score_team1 = 0
    score_team2 = 0
    possesion_counter = 0
    yds_to_goal = 80
    while possesion_counter <= possessions:
        change_possession = False
        # obs = env.reset()
        state = env.reset(yds_to_goal=yds_to_goal)
        print('possesion', possesion_counter + 1)
        while change_possession != True:
            if show_states == 'y':
                print('down:', state[1] + 1)
                print('yds to 1st', state[2])
                print('yds to goal', state[0])
            # Q_values = model.predict([state], verbose=0)[0]
            # action = Q_values.argmax()  # optimal action according to the DQN
            if possesion_counter % 2 == 0:
                Q_values = model.predict([state], verbose=0)[0]
                action = Q_values.argmax()  # optimal action according to the DQN
            else:
                if team2_strat == 'simp':
                    action = simple_strat(state)
                elif team2_strat == 'nn':
                    Q_values = model2.predict([state], verbose=0)[0]
                    action = Q_values.argmax()  # optimal action according to the DQN
            next_state, reward, done, info, yds, score = env.step(action)
            state = next_state
            if show_states == 'y':
                print('action', action)
                print('yds', yds)
            if done:
                # even
                if possesion_counter % 2 == 0:
                    score_team1 += score
                else:
                    score_team2 += score
                print()
                print('team 1:', score_team1, 'team 2:', score_team2)
                # if score assume touchback for other team
                if state[0] == 0:
                    yds_to_goal = 80
                else:
                    yds_to_goal = 100 - state[0]
                change_possession = True
        possesion_counter += 1


# options dqn, fixed_dqn, double_dqn
rl_type = 'fixed_dqn'

# calc_mean_sd()

env = FootballEnv()

# not sure the below is needed
# https: // www.tensorflow.org / api_docs / python / tf / keras / backend / clear_session
tf.keras.backend.clear_session()
# not sure if above requires below
tf.random.set_seed(seed)
replay_buffer = deque(maxlen=2000)
# episodes = 2_000


def exe(episodes=1_000, batch_size=32, discount_factor=.95, target_update_freq=50):
    best_weights, rewards = train(episodes=episodes, batch_size=batch_size, discount_factor=discount_factor
                                  , target_update_freq=target_update_freq, ep_ticker='y', ep_chart='n')
    model.set_weights(best_weights)  # extra code – restores the best model weights
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model.save_weights('./playCall_weights/weights_{episodes}'.format(episodes=episodes))
    avg_return = test(episodes=100, show_states='n')
    return avg_return


# rl_types = ['dqn', 'fixed_dqn', 'double_dqn']
# batch_sizes = [32, 64]
# discount_factors = [.93, .95, .97]
# ep_counts = [1_000, 2_000, 3_000]
# update_freqs = [50, 100]
# rl_types = ['fixed_dqn']
# batch_sizes = [32]
# discount_factors = [.95]
# ep_counts = [1_000, 2_000]
# update_freqs = [50]
#
# rl_type_list = []
# ep_count_list = []
# batch_size_list = []
# discount_factor_list = []
# avg_return_list = []
# update_freq_list = []
# for rl_type in rl_types:
#     for batch_size in batch_sizes:
#         for discount_factor in discount_factors:
#             for ep_count in ep_counts:
#                 # this doesn't impact standard dqn, only fixed and double
#                 for update_freq in update_freqs:
#                     print(rl_type)
#                     print(batch_size)
#                     print(discount_factor)
#                     print(ep_count)
#                     print(update_freq)
#                     print()
#                     model = create_model()
#                     if (rl_type == 'fixed_dqn') | (rl_type == 'double_dqn'):
#                         target = tf.keras.models.clone_model(model)  # clone the model's architecture
#                         target.set_weights(model.get_weights())  # copy the weights
#                     avg_return = exe(episodes=ep_count, batch_size=batch_size, discount_factor=discount_factor
#                                      , target_update_freq=update_freq)
#                     rl_type_list.append(rl_type)
#                     ep_count_list.append(ep_count)
#                     batch_size_list.append(batch_size)
#                     discount_factor_list.append(discount_factor)
#                     avg_return_list.append(avg_return)
#                     update_freq_list.append(update_freq)
#
# df = pd.DataFrame({'rl_type': rl_type_list, 'ep_count': ep_count_list, 'batch_size': batch_size_list
#                       , 'discount_factor': discount_factor_list, 'avg_return': avg_return_list, 'update_freq': update_freq_list})
# df.to_csv('rl_summary.csv')

# play rl vs random strat
model = create_model()
model.load_weights('./playCall_weights/weights_{episodes}'.format(episodes=2_000))
play(possessions=100, show_states='n')

# play rl vs rl
model = create_model()
model.load_weights('./playCall_weights/weights_{episodes}'.format(episodes=2_000))
model2 = create_model()
model2.load_weights('./playCall_weights/weights_{episodes}'.format(episodes=1_000))
play(possessions=100, team2_strat='nn', show_states='n')