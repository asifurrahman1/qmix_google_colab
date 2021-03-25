import numpy as np
from matplotlib import pyplot 
import os
import sys
sys.path.insert(0,"/root/qmix")

from rollout import RolloutWorker  
from agent import Agents 
from adversary import Adversary
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.count = 1
        self.agents = Agents(args)
        self.adversarial = Adversary(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args, self.adversarial)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.expert_data = []
        self.adv_data =[]
        self.episode = []
        self.q_diff = []
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map_name
        self.save_path1 = '/gdrive/MyDrive/Colab Notebooks/Saved_data/result' + '/' + args.alg + '/' + args.map_name+'/attack_data'
        self.save_path1 = self.save_path1+'/'+self.args.attack_name+'/'+'atk_rate_{}'.format(self.args.attack_rate)
        
        self.save_path2 = '/gdrive/MyDrive/Colab Notebooks/Saved_data/result' + '/' + args.alg + '/' + args.map_name+'/normal'
        # if not os.path.exists(self.save_path2):
        #   os.makedirs(self.save_path2)
    def savedata(self, data):
      if not os.path.exists(self.save_path2):
        os.makedirs(self.save_path2)
      np.save(self.save_path2 + '/episode_data', data)
      print('saved_called')

    # def add_tractectory(self, data):
    #     data_set.append(data)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        data = []
        while time_steps < self.args.n_steps:
            #print('Run {}, time_steps {}'.format(num, time_steps))
            
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                print('Time_steps {}:'.format(time_steps),"Win rate:", win_rate,"Episode reward:", episode_reward)
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
                if evaluate_steps>2:
                  self.savedata(data)
            d_list = []
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps,t_data,_,_ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                v = t_data.copy()
                time_steps += steps
                if evaluate_steps>2:
                  d_list.append(v)
                # print(_)
            if evaluate_steps>2:
              data.append(d_list)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.buffer.store_episode(episode_batch)
            if evaluate_steps <2:
              for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1

        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)
        
        
    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        data = []
        for epoch in range(self.args.evaluate_epoch):
            #============>
            self.episode, episode_reward, win_tag, _, t_data, self.adv_data, self.q_diff = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            d_data = t_data.copy()
            data.append(d_data) 
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        # self.savedata()
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch
    
  
    def plt(self, num):
        # plt.figure()
        # plt.ylim([0, 120])
        # plt.cla()
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('Win rates')
        # plt.subplot(2, 1, 2)
        # plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('episode_rewards')
        fig = pyplot.figure()
        fig.subplots_adjust(hspace=0.5)
        # fig.ylim([0, 120])
        # fig.cla()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(range(len(self.win_rates)), self.win_rates)
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax1.set_ylabel('Win rates')

        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(range(len(self.episode_rewards)), self.episode_rewards)
        ax2.set_ylim([0, 50])
        ax2.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax2.set_ylabel('Episode Rewards')

        #====================================================
        fig2 = pyplot.figure()
        fig2.subplots_adjust(hspace=0.5)
      
        ax1 = fig2.add_subplot(3,1,1)
        ax1.plot(range(len(self.q_diff)), self.q_diff)
        ax1.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax1.set_ylabel('Max-Min: Qvalue')
        
        ax2 = fig2.add_subplot(3,1,2)
        ax2.plot(range(len(self.win_rates)), self.win_rates)
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax2.set_ylabel('Win rates')

        ax3 = fig2.add_subplot(3,1,3)
        ax3.plot(range(len(self.episode_rewards)), self.episode_rewards)
        ax3.set_ylim([0, 50])
        ax3.set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        ax3.set_ylabel('Episode Rewards')
        #====================================================
        if self.args.adversary:
          if not os.path.exists(self.save_path1):
            os.makedirs(self.save_path1)
          text = 'adversary' 
          fig.savefig(self.save_path1 + '/plt_{}.png'.format(text), format='png')
          np.save(self.save_path1 + '/win_rates_{}'.format(text), self.win_rates)
          np.save(self.save_path1 + '/episode_rewards_{}'.format(text), self.episode_rewards)
          np.save(self.save_path1 + '/data_set', self.data_set)
          np.save(self.save_path1 + '/adv_data', self.adv_data)
          if self.args.attack_name == 'strategic':
            fig2.savefig(self.save_path1 + '/max_min_diff_plt_{}.png', format='png')
        else:
          if not os.path.exists(self.save_path2):
            os.makedirs(self.save_path2)
          text = 'normal'
          fig.savefig(self.save_path2 + '/plt_{}.png'.format(text), format='png')
          # np.save(self.save_path2 + '/episode_data', self.data_set)
          np.save(self.save_path2 + '/win_rates_{}'.format(text), self.win_rates)
          np.save(self.save_path2 + '/episode_rewards_{}'.format(text), self.episode_rewards)
          # fig.close()
          # fig2.close()
