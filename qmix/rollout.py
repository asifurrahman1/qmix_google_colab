import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from numpy import save
import sys
sys.path.insert(0,"/root/qmix")


class RolloutWorker:
    def __init__(self, env, agents, args, adversarial):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.adversarial = adversarial
        
        print('Successful Init of RolloutWorker')
    #===============>entry 1
  

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        a, adv_o, adv_action, threshold = [], [], [], []
        
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        #==================================================
        #==============HIDDEN INITIALIZED =================
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                #print("Agent ID=",  agent_id)
                avail_action = self.env.get_avail_agent_actions(agent_id)
                #===========================================================
                #=========IF STATE PERTURBATION CHANGE HERE ================
                #===========================================================
           
                if self.args.victim_agent == agent_id and self.args.adversary and np.random.uniform() <= self.args.attack_rate:
                  #USE ADVERSARIAL ATTACK HERE
                  #print("Attack launched")
                  if self.args.attack_name == "random":
                    action = self.adversarial.random_attack(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                  elif self.args.attack_name == "random_time":
                    #self.adversarial.policy.init_hidden(1)
                    q_val = self.agents.get_qvalue(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    action = self.adversarial.random_time_attack(q_val, avail_action)
                    #print("attack successful")
                  elif self.args.attack_name == "strategic":
                    demo_thrs = 0.5
                    q_val = self.agents.get_qvalue(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    action, diff = self.adversarial.strategic_time_attack(q_val, avail_action, demo_thrs)
                    threshold.append(diff)
                    # action = self.agents.choose_strategic_action(obs[agent_id], last_action[agent_id], agent_id,
                    #                                    avail_action, epsilon, evaluate)
                    #action = self.adversarial_policy(obs[agent_id],avail_action)
                else:
                  action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
                if agent_id == self.args.victim_agent:
                  adv_o.append(obs[agent_id])
                  adv_action.append(action)
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            a.append(actions)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])
            #a.append(np.zeros((self.n_agents, self.n_actions)))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
    
        #===========EPISODE SAVED FOR CAM=======================
        data_set = dict(
                        state = s.copy(),
                        observation = o.copy(),
                        action = a.copy(),
                        next_state = s_next.copy(),
                        observation_next=o_next.copy(),
                        )
      
        adv_data = dict(
                        adv_observation = adv_o.copy(),
                        adv_action = adv_action.copy()
                      )
        
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
   
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step, data_set, adv_data, threshold

