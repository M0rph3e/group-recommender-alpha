import numpy as np
import pandas as pd
import torch 

from config import Config
from agent import DDPGAgent
from env import Env
import random

class Offline(object):
    """
    Offline data generation
    """
    def __init__(self, config : Config):
        self.item_num = config.item_num
        self.group_num= config.group_num
        self.action_size = config.action_size
        self.history_length = config.history_length

    def random_state(self):
        """
        generate random [group,states] during offline rollouts
        :return state (list) : group + K-item or list of action depending on config.action_size
        """
        state = [np.random.randint(1,self.group_num)] + (random.sample(range(1,self.item_num),self.history_length)) #+ to concat
        return state

    def random_reward(self):
        """
        generate random reward during offline rollouts
        :return reward (int) : randomly output 0 or 1
        """
        return random.randint(0,1)

    def off_step(self, action : int):
        """
        offline policy during one training step
        :params: action (int) : action of the agent 
        :return: new_state, reward, done, info
        """
        state = self.random_state()
        group_id = state[0]
        history = state[1:] 

        reward = self.random_reward()

        if reward > 0:
            history = history[1:] + [action]

        new_state = [group_id] + history
        self.state = new_state
        done = False
        info = {}

        return new_state, reward, done, info


    def train_offline(self, agent: DDPGAgent,config : Config,
          df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame()):
        """
        Train the agent offline without interacting with the environment
        :param config : configuration file
        :param agent: agent
        :param evaluator: evaluator
        :param df_eval_user: user evaluation data
        :param df_eval_group: group evaluation data
        :return agent : agent trained offline so it can be deployed in the environment during training
        """
        rewards = []
        for episode in range(config.offline_episodes):
            state = self.random_state()
            agent.noise.reset()
            episode_reward = 0

            for step in range(config.num_steps):
                action = agent.get_action(state,with_noise=True)
                new_state, reward, _, _ = self.off_step(action)
                agent.replay_memory.push((state, action, reward, new_state)) #add to buffer
                state = new_state
                episode_reward += reward

                if len(agent.replay_memory) >= config.batch_size:
                    agent.update()

            rewards.append(episode_reward / config.num_steps)
            print('Offine Episode = %d, average reward = %.4f' % (episode, episode_reward / config.num_steps))
            
            #clear buffer before pytting online
            agent.replay_memory.buffer.clear()
        return agent
