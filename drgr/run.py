import main
import copy
from offline import Offline
from config import Config
from agent import DDPGAgent
from data import DataLoader
from env import Env
from eval import Evaluator
from utils import OUNoise

if __name__ == "__main__":
    experiment = "previous" # ["baseline","random","famous","previous"] 
    NUM_ITER = 5
    config_base = Config()
    config = config_base#copy.copy(config_base)
    config.group_name=experiment+" new-trial"
    if experiment == "baseline":
        config.is_offline=False
    else:
        config.is_offline=True
        config.offline_policy=experiment
    for n in range(NUM_ITER): #train many times
        #inits     
        dataloader = DataLoader(config)
        rating_matrix_train = dataloader.load_rating_matrix(dataset_name='val')
        df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
        df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test') 
        env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='val')
        noise = OUNoise(config=config)
        agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=False)
        evaluator = Evaluator(config=config)
        offline = Offline(config=config,rating_matrix=rating_matrix_train)
        print('-'*20)
        print("Training with Seed :"+str(config.seed)+" for experiment :" + str(config.num))
        print('-'*20)

        #usual train call from main
        if config.is_offline: #train offline ?
            offline_agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=False)
            if config.generate_offline_data : # regenerate offline data ?
                offline.get_offline_data(policy=config.offline_policy)
            offline_agent = offline.train_offline(agent=offline_agent, evaluator=evaluator,
                                    df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test,save_agent=config.save_agent,policy=config.offline_policy)
            agent = offline.copy_agent(offline_agent,agent)
        if config.is_off_policy:
            main.train(config=config, env=env, agent=agent, evaluator=evaluator,
            df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
            
        #increment seed and project num
        config.seed+=1
        config.num+=1

