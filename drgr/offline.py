import numpy as np
import pandas as pd
import torch 

import os
from config import Config
from agent import DDPGAgent
from eval import Evaluator
from env import Env
import random
import pickle
import wandb
from scipy.sparse.csr import csr_matrix
from sklearn import preprocessing as pre
from sklearn.decomposition import NMF


class Offline(object):
    """
    Offline data generation
    """
    def __init__(self, config : Config, rating_matrix : csr_matrix):
        self.config = config
        #set seed for reproducibility
        np.random.seed(config.seed)
        random.seed(config.seed)

        #offline data directory 
        if not os.path.exists(self.config.offline_path):
            os.mkdir(self.config.offline_path)
        
        # get rating matrix data (basically same principle from env)
        self.rating_matrix = rating_matrix
        rating_matrix_coo = rating_matrix.tocoo()
        rating_matrix_rows = rating_matrix_coo.row
        rating_matrix_columns = rating_matrix_coo.col
        self.rating_matrix_index_set = set(zip(*(rating_matrix_rows, rating_matrix_columns)))
        self.rating_matrix_pred = None

        if self.config.offline_policy == 'famous': # if famous policy we use NMF with rating matrix
            matrix_name = 'mat' +'_'+ 'offline' + '_' + str(self.config.env_n_components) + '.npy'
            self.pred_matrix_path = os.path.join(self.config.offline_path,matrix_name)
            self.rating_matrix_pred = self._get_pred_matrix()
            self.ranking = self._get_famous_id(k=self.config.k_famous)

    def _get_pred_matrix(self):
        """
        get prediction matrix proba using NMF
        :return: rating_matrix_pred
        """
        #instanciate model (same as env)
        if not os.path.exists(self.pred_matrix_path):
            model = NMF(n_components=self.config.env_n_components, init='random', tol=self.config.env_tol,
                            max_iter=self.config.env_max_iter, alpha=self.config.env_alpha, verbose=True,
                            random_state=0)
            print('-' * 50)
            print('Train NMF:')
            W = model.fit_transform(X=self.rating_matrix)
            H = model.components_ 
            #normalizing the rating matrix pred in this case to have 
            rating_matrix_pred = pre.MinMaxScaler().fit_transform(W @ H)
            print('-' * 50)
            np.save(self.pred_matrix_path, rating_matrix_pred)
            print('Save rating matrix pred:', self.pred_matrix_path)
        else:
            rating_matrix_pred = np.load(self.pred_matrix_path)
            print('Load rating matrix pred:', self.pred_matrix_path)

        return rating_matrix_pred


    
    def random_state(self):
        """
        generate random [group,states] during offline rollouts
        :return state (list) : group + K-item or list of action depending on config.action_size
        """
        state = [np.random.randint(1,self.config.total_group_num)] + (random.sample(range(1,self.config.item_num),self.config.history_length)) #+ to concat
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
        buffer = []
        for i in range(self.config.offline_data_size):
            if policy == 'random' :
                buffer.append(self.random_policy())
            elif policy=='famous':
                buffer.append(self.famous_policy())        
        #dump generated data in pkl
        if policy!= 'previous':
            with open(offline_save_path,'wb') as file:
                pickle.dump(buffer,file)
        print("Done")
        return buffer



    def random_policy(self):
        """
        random offline policy during one training step
        :params: action (int) : action of the agent 
        :return: state, action reward, new_state
        """
        state = self.random_state()
        group_id = state[0]
        history = state[1:] 
        #print("(group_id: " + str(group_id) + " history: " + str(history) +")")
        action = self.random_action()
        reward = self.random_reward()

        if reward > 0:
            history = history[1:] + [action]

        new_state = [group_id] + history

        return state, action, reward, new_state

    def famous_policy(self):
        """
        offline policy during one training step recommanding the most rated items in the rating matrix
        :params: action (int) : action of the agent 
        :return: state, action reward, new_state
        """
        #TO DO 
        #Generate the minimized rating matrix with most famous movies
        ranking = self.ranking
        state = [np.random.randint(1,self.config.total_group_num)]  + np.random.choice(ranking,size=self.config.history_length).tolist()
        action = np.random.choice(ranking)


        
        #get rewrd with NMF rating matrix pred
        group_id = state[0]
        history = state[1:]
        #print("(group_id: " + str(group_id) + " history: " + str(history) +")")
        if (group_id, action) in self.rating_matrix_index_set:
            reward = self.rating_matrix[group_id, action]
        else:
            reward_probability = self.rating_matrix_pred[group_id, action]
            reward = np.random.choice(self.config.rewards, p=[1 - reward_probability, reward_probability])

        if reward > 0:
            history = history[1:] + [action]
        
        new_state = [group_id] + history
        
        return state, action, reward, new_state
    
    def _get_famous_id(self,k: int):
        """
        utility fctn to return the reduced rating matrix 
        :params: k (int) number of "most rated movies to output"
        :return: ranking (np array)
        """
        r = []
        for i in range(self.rating_matrix.shape[1]):
            r.append(self.rating_matrix.getcol(i).count_nonzero())
            #here `count_nonzero` counts `rating==1` where `getnnz` counts non-missing values (0s and 1s), so can be changed if we wanted most rated or the best rated
        r = np.array(r)
        # from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        ranking = np.argpartition(r, -k)[-k:]
        return ranking

    def copy_agent(self, agent:DDPGAgent, agent_target:DDPGAgent):
        """
        Copy Actor and Critic network from a source agent to target DDPG agent
        :param agent: source agent
        :param agent_target: target_agent
        :return agent_target: 
        """
        agent_target.actor.load_state_dict(agent.actor.state_dict())
        agent_target.critic.load_state_dict(agent.critic.state_dict())

        return agent_target

    def get_weights(self, agent:DDPGAgent):
        """
        Copy Actor and Critic weigths from agent and return them
        :param agent: source agent
        :return agent weight: 
        """
        weight = agent.actor.state_dict().copy() , agent.critic.state_dict().copy()
        return weight


    def train_offline(self, evaluator:Evaluator,agent: DDPGAgent,
                      df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame(),policy='random',reload=False,save_agent=True):
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
        if policy =='previous':
            #from https://stackoverflow.com/questions/15312953/choose-a-file-starting-with-a-given-string
            prefix = [filename for filename in os.listdir(self.config.offline_path) if filename.startswith(self.config.offline_policy)]
            offline_save_path = os.path.join(self.config.offline_path,prefix.pop()) #pop to get at least one previous buffer (might change later)
            print(offline_save_path)
        else:
            offline_save_path = os.path.join(self.config.offline_path, policy + '_' + 
                                            str(self.config.offline_data_size) + '.pkl')
        

        agent_save_path = os.path.join(self.config.offline_path, policy + '_' + 
                                            str(self.config.offline_data_size) + 'agent.pkl')
        with wandb.init(project=self.config.project, entity= self.config.entity,job_type="train", name=self.config.name+'_offline', group='offline-'+self.config.group_name) as run:
            #load historical data buffer     
            if not os.path.exists(offline_save_path) or reload:
                buffer = self.get_offline_data(policy=policy)
            else:
                with open(offline_save_path, 'rb') as file:
                    buffer = pickle.load(file)

            best_top_k = np.NINF if self.config.keep_best else None #negative infinity to be sure to update a t+1
            for step in range(self.config.offline_step):
                #put historical data batch to agent
                batch = random.sample(buffer,k=self.config.offline_batch_size)
                for d in batch :
                    agent.replay_memory.push(d)
                #update the agent on historical buffer data
                if len(agent.replay_memory) >= self.config.offline_batch_size:
                    save_weight = self.get_weights(agent) if self.config.keep_best else None
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
                #AFTER EVAL, check if evaluation gives better top K value (for time constraint we'll check top_20 recall progress, but curves look alike in the online env)
                    recall_score_user_20,_ = evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=20)
                    #recall_score_user_20 = avg_recall_score_user # get the last value in for loop so here for Top_20
                    if self.config.keep_best:
                        if (best_top_k<=recall_score_user_20): #if improvement change best value
                            best_top_k=recall_score_user_20
                        else: #else keep previous weights
                            agent.actor.load_state_dict(save_weight[0]),agent.critic.load_state_dict(save_weight[1]) # load weights of previous best perf


        #dump pretrained model
        if save_agent:
            with open(agent_save_path,'wb') as file:
                pickle.dump(agent,file)
            print("Save agent at " + agent_save_path)
        return agent
