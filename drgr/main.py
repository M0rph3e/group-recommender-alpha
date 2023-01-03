"""
Main
"""
import pandas as pd

from agent import DDPGAgent
from config import Config
from data import DataLoader
from env import Env
from eval import Evaluator
from utils import OUNoise
from offline import Offline
import wandb

def train(config: Config, env: Env, agent: DDPGAgent, evaluator: Evaluator,
          df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame()):
    """
    Train the agent with the environment

    :param config: configurations
    :param env: environment
    :param agent: agent
    :param evaluator: evaluator
    :param df_eval_user: user evaluation data
    :param df_eval_group: group evaluation data
    :return:
    """
    rewards = []
    with wandb.init(project=config.project, entity= config.entity,job_type="train", name=config.name) as run:
        for episode in range(config.num_episodes):
            state = env.reset()
            agent.noise.reset()
            episode_reward = 0

            for step in range(config.num_steps):
                action = agent.get_action(state, with_noise=config.with_noise)
                new_state, reward, _, _ = env.step(action)
                agent.replay_memory.push((state, action, reward, new_state))
                state = new_state
                episode_reward += reward

                if len(agent.replay_memory) >= config.batch_size:
                    agent.update()

            rewards.append(episode_reward / config.num_steps)
            print('Episode = %d, average reward = %.4f' % (episode, episode_reward / config.num_steps))

            #LOG INTO WANDB
            wandb.log({"episodes":episode, "average reward": (episode_reward / config.num_steps)})
    
            if (episode + 1) % config.eval_per_iter == 0:
                for top_K in config.top_K_list:
                    avg_recall_score_user,avg_ndcg_score_user = evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=top_K)
                    #log to WANDB
                    wandb.log({"Average Recall@"+str(top_K)+" Score for user":avg_recall_score_user,
                                "average NDCG@"+str(top_K)+" Score for user": avg_ndcg_score_user,"Episode" : episode})

                for top_K in config.top_K_list:
                    avg_recall_score_goup, avg_ndcg_score_group = evaluator.evaluate(agent=agent, df_eval=df_eval_group, mode='group', top_K=top_K)
                    #log to WANDB
                    wandb.log({"Average Recall@"+str(top_K)+" Score for Group":avg_recall_score_goup,
                                "average NDCG@"+str(top_K)+" Score for Group": avg_ndcg_score_group,"Episode" : episode})


if __name__ == '__main__':
    config = Config()
    dataloader = DataLoader(config)
    rating_matrix_train = dataloader.load_rating_matrix(dataset_name='val')
    df_eval_user_test = dataloader.load_eval_data(mode='user', dataset_name='test')
    df_eval_group_test = dataloader.load_eval_data(mode='group', dataset_name='test')
    env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='val')
    noise = OUNoise(config=config)
    agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
    evaluator = Evaluator(config=config)
    offline = Offline(config=config,rating_matrix=rating_matrix_train)
    if config.is_offline: #train offline ?
        offline_agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=False)
        if config.generate_offline_data : # regenerate offline data ?
            offline.get_offline_data(policy=config.offline_policy)
        offline_agent = offline.train_offline(agent=offline_agent, evaluator=evaluator,
                                      df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test,save_agent=config.save_agent)
        agent = offline.copy_agent(offline_agent,agent)
    if config.is_off_policy:
        train(config=config, env=env, agent=agent, evaluator=evaluator,
              df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
