import numpy as np
import pandas as pd
import torch 

import os
from config import Config
from agent import DDPGAgent
from eval import Evaluator
from env import Env
from collections import deque
import random
import pickle
import wandb

class Offline(object):
    """
    Offline data generation
    """
    def __init__(self, config : Config):
        self.config = config
        #set random seed for reproducibility
        np.random.seed(config.seed)

        #offline data directory 
        if not os.path.exists(self.config.offline_path):
            os.mkdir(self.config.offline_path)

    def random_state(self):
        """
        generate random [group,states] during offline rollouts
        :return state (list) : group + K-item or list of action depending on config.action_size
        """
        state = [np.random.randint(1,self.config.group_num)] + (random.sample(range(1,self.config.item_num),self.config.history_length)) #+ to concat
        return state

    def random_action(self):
        """
        generate random action (item) during offline rollouts
        :return action (int) : item to recommend in a transition
        """
        return np.random.randint(1,self.config.item_num)

    def random_reward(self):
        """
        generate random reward during offline rollouts
        :return reward (int) : randomly output 0 or 1
        """
        return random.randint(0,1)

    def get_offline_data(self, policy="random"):
        """
        generate offline transition data 
        :param config: configuration file:
        :param policy: in ["random","famous"] : policy to simulate historical data (only random actually)
        :return offline data buffer: np array simaliting dequeu in format ([state, action, reward, new_state],...) 
        """
        offline_save_path = os.path.join(self.config.offline_path, policy + '_' + 
                                            str(self.config.offline_data_size) + '.pkl')
        print("Generating offline data with " + policy + " policy")
        buffer = deque([])
        for i in range(self.config.offline_data_size):
            if policy == 'random' :
                buffer.append(self.random_policy())
            elif policy=='famous':
                pass #TO DO

        #dump genrated data in pkl
        with open(offline_save_path,'wb') as file:
            pickle.dump(buffer,file)

        return buffer



    def random_policy(self):
        """
        offline policy during one training step
        :params: action (int) : action of the agent 
        :return: state, action reward, new_state
        """
        state = self.random_state()
        group_id = state[0]
        history = state[1:] 
        action = self.random_action()
        reward = self.random_reward()

        if reward > 0:
            history = history[1:] + [action]

        new_state = [group_id] + history

        return state, action, reward, new_state


    def train_offline(self, evaluator:Evaluator,agent: DDPGAgent,
                      df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame(),policy='random',reload=False):
        """
        Train the agent offline without interacting with the environment
        :param config : configuration file
        :param agent: agent
        :param evaluator: evaluator
        :param df_eval_user: user evaluation data
        :param df_eval_group: group evaluation data
        :return agent : agent trained offline so it can be deployed in the environment during training
        """
        print("Training offline")
        offline_save_path = os.path.join(self.config.offline_path, policy + '_' + 
                                            str(self.config.offline_data_size) + '.pkl')
        with wandb.init(project=self.config.project, entity= self.config.entity,job_type="train", name=self.config.name+'_offline') as run:
            #load historical data buffer     
            if not os.path.exists(offline_save_path) or reload:
                buffer = self.get_offline_data(policy=policy)
            else:
                with open(offline_save_path, 'rb') as file:
                    buffer = pickle.load(file)

            for step in range(self.config.offline_step):
                #put historical data to agent 
                agent.replay_memory.push(buffer.pop())
                if len(buffer)==0:
                    print("UNROLLED ALL OFFLINE DATA")
                    break
                #update the agent on historical buffer data
                if len(agent.replay_memory) >= self.config.batch_size:
                    agent.update()

                #Evaluate agent each `offline_eval_per_step` 
                if (step + 1) % self.config.offline_eval_per_step == 0:
                    for top_K in self.config.top_K_list:
                        avg_recall_score_user,avg_ndcg_score_user = evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=top_K)
                        #log to WANDB
                        wandb.log({"Average Recall@"+str(top_K)+" Score for user":avg_recall_score_user, "average NDCG@"+str(top_K)+" Score for user": avg_ndcg_score_user},step=step)

                    for top_K in self.config.top_K_list:
                        avg_recall_score_goup, avg_ndcg_score_group = evaluator.evaluate(agent=agent, df_eval=df_eval_group, mode='group', top_K=top_K)
                        #log to WANDB
                        wandb.log({"Average Recall@"+str(top_K)+" Score for Group":avg_recall_score_goup, "average NDCG@"+str(top_K)+" Score for Group": avg_ndcg_score_group},step=step)

            
        return agent
