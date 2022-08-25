import numpy as np
import torch
from common.ExperienceReplay import ExperienceReplay
from common.TimeCount import TimeCount
from common.math import *
from common.utils import Logger
from matplotlib import pyplot as plt
from network.BaseNet import FullyConnected
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QLearning(object):
    def __init__(self, env, model_arg, env_arg):
        self.env = env
        self.model_arg, self.env_arg = model_arg, env_arg

        # logger parameter
        self.train_log = Logger()
        self.evaluate_log = Logger()

        # TODO add beliefstate model into model arg

        # set parameters
        for key, value in env_arg.items():
            setattr(self, key, value)

        for key, value in model_arg.items():
            setattr(self, key, value)


        self.initialize()


    def initialize(self):
        # initialize experience replay
        er_arg = self.env_arg.copy()
        er_arg['er_size'] = self.er_size
        self.experience_replay = ExperienceReplay(er_arg)

        # TODO check episode_limit
        # initialize model
        self.model = FullyConnected(self.state_dim, self.action_dim).to(DEVICE)
        # self.model = FullyConnected(400, self.action_dim).to(DEVICE)
        self.target_model = FullyConnected(self.state_dim, self.action_dim).to(DEVICE)
        # self.target_model = FullyConnected(400, self.action_dim).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # initialize epsilon
        self.epsilon_high, self.epsilon_low = 1, 0.01
        self.epsilon = self.epsilon_high

        # time count
        self.time_count = TimeCount()
        self.time_count.reset()
        self.step_count = 0

    def learn(self, ):
        # self.showEpsilon()
        self.training = True
        if self.buffer_ini:
            # fill experience replay
            for i_ep in tqdm(range(self.er_size // self.episode_limit)):
                self.generateEpisode()

        range_run_episode = tqdm(range(self.n_ep_training))
        for i in range_run_episode:
            self.ep_count = i
            self.runEp(log=self.train_log)

            # print out episode details
            log = self.train_log
            range_run_episode.set_description('epoch: {}. return: {} at time {} with epsilon {:.2f}'.format(self.ep_count,
                np.round(log.getCurrent('reward')), log.getCurrent('episode_len'), self.epsilon))

    def runEp(self, log=None):
        self.generateEpisode()
        # update the same as number of steps
        for i_step in range(self.episode_limit):
            self.update()

        # update models
        if self.ep_count % self.target_update_cycle == 0 and self.ep_count > 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # append log
        dic_item = {
            'loss': self.loss.item(),
            'reward': int(self.episode_reward),
            'episode_len': int(self.episode_end_time),
            'agent_start': self.env.agent.start_position.tolist(),
            'agent_end': self.env.agent.position.tolist(),
            'time_process': float(self.time_count.count())
        }
        if log is not None:
            log.addListItem(dic_item)



        # evalute model
        # TODO add save evaluate log
        if self.evaluate_model:
            if (self.ep_count + 1) % self.save_interval == 0 or (self.ep_count == self.n_ep_training - 1) or (
                    self.ep_count == 0):
                self.evaluateModel()

        # check save model
        if self.save_model:
            if (self.ep_count + 1) % self.save_interval == 0 or (self.ep_count == self.n_ep_training - 1) or (
                    self.ep_count == 0):
                self.saveLog()

    def evaluateModel(self):
        def getEvaluateData(n_episode, use_epsilon=True):
            source_pos, agent_action, agent_pos, water_flow_force, done = [], [], [], [], []
            for i in tqdm(range(n_episode)):
                self.generateEpisode(use_epsilon=use_epsilon, evaluate=True)
                source_pos.append(self.episode_source_pos)
                agent_action.append(self.episode_agent_action)
                agent_pos.append(self.episode_agent_pos)
                water_flow_force.append(self.episode_water_flow_force)
                done.append(self.episode_done)

            return np.array(source_pos), np.array(agent_action), np.array(agent_pos), np.array(water_flow_force), np.array(done).astype(np.bool)

        # use epsilon map

        # TODO add evalute episode
        # print("evaluating model")

        # not use epsilon map
        source_pos, agent_action, agent_pos, water_flow_force, done = getEvaluateData(self.n_ep_evaluate,
                                                                                      use_epsilon=False)
        agent_centric_map, source_centric_map, water_centric_map = processMap(agent_pos, source_pos,
                                                                                  water_flow_force, done)

        #use epsilon
        source_pos, agent_action, agent_pos, water_flow_force, done = getEvaluateData(self.n_ep_evaluate)
        epsilon_agent_centric_map, epsilon_source_centric_map, epsilon_water_centric_map = processMap(agent_pos,
                                                                                                       source_pos,
                                                                                                       water_flow_force, done)



        self.evaluate_dic_item = {
            'agent_centric_map': agent_centric_map,
            'source_centric_map': source_centric_map,
            'water_centric_map': water_centric_map,
            'agent_centric_epsilon_map': epsilon_agent_centric_map,
            'source_centric_epsilon_map': epsilon_source_centric_map,
            'water_centric_epsilon_map': epsilon_water_centric_map,
            'epsilon': self.epsilon,
            'ep_count': self.ep_count,

        }
        self.evaluate_log.addListItem(self.evaluate_dic_item)

    def reset(self):
        pass

    def getAct(self, state, use_epsilon=True):

        bsize = state.shape[0]
        action_prob = self.model(state.to(DEVICE))

        prob, action = action_prob.max(dim=-1)

        # check e-greedy
        if use_epsilon:
            use_greedy = (torch.rand(bsize) < self.epsilon).to(torch.int).to(DEVICE)
            # action_greedy = torch.randint(self.action_dim, (bsize,)).to(DEVICE)
            action_greedy = torch.tensor(self.env.agent.getStochasticAction()).reshape(1).to(DEVICE)
            pick_action = torch.where(use_greedy == 1, action_greedy, action)
        else:
            pick_action = action
        return pick_action  # bsize, 1

    def encodeAction(self, action):
        return torch.eye(self.action_dim)[action]


    def getState(self, obs, action):
        state = self.belief_state_model(obs[:, None, :].to(DEVICE), action[None, None, :].to(DEVICE), fresh=False)
        # state, self.hidden = self.belief_state_model(obs[:, None, :].to(DEVICE),
        #                                              self.encodeAction(action)[:, None, :].to(DEVICE), self.hidden)
        return state
    # def getBeliefState(self, obs, action, hidden):
    #     # obs = (bsize, time_len, feature_dim)
    #     # action = (bsize, time_len, action_dim)
    #     # hidden = (bsize, mem_size, mem_size, feature_dim)
    #     return self.belief_state_model(obs, action, hidden)

    def generateEpisode(self, use_epsilon=True, evaluate=False):
        # reset policy
        self.reset()
        bsize = 1
        self.hidden = self.belief_state_model.hiddenInitialize(bsize=bsize)
        self.belief_state_model.hiddenInitialize(bsize=bsize)
        obs = self.env.reset()

        self.episode_agent_pos = [self.env.agent.position.tolist()]
        self.episode_water_flow_force = self.env.wind.water_flow_force.tolist()
        self.episode_source_pos = self.env.source.position.tolist()
        self.episode_agent_action = []
        self.episode_done = [0]
        # reset environment

        # TODO retrain source localize model with initialize action

        action = torch.randint(self.action_dim, (bsize,))
        # state = self.belief_state_model(obs[:, None, :].to(DEVICE), action[None, None, :].to(DEVICE), fresh=False)
        state = self.getState(obs, action)
                                                     #self.encodeAction(action)[:, None, :].to(DEVICE))#, hidden)
        state = state.detach().squeeze(dim=1).to('cpu')

        # initlize saved array
        observation_episode, action_episode, reward_episode, done_episode = [], [], [], []
        for i_step in range(self.episode_limit):
            # step environment
            action = self.getAct(state, use_epsilon=use_epsilon)
            obs_next, reward, done, info = self.env.step(action)

            # process belief state
            # state_next = self.belief_state_model(obs_next[:, None, :].to(DEVICE), action[None, None, :].to(DEVICE), fresh=False)

            state_next = self.getState(obs_next, action)
                                                         #self.encodeAction(action)[:, None, :].to(DEVICE))#, hidden)
            state_next = state_next.detach().squeeze(dim=1).to('cpu')  # bsize, hid_dim

            # prepare experience save
            step = {
                's': state,
                'a': action,
                'r': reward,
                's_next': state_next,
                'done': done,
            }

            if evaluate == False:
                self.experience_replay.storeEpisode(step)
            reward_episode.append(reward)

            # update obs
            state = state_next
            if done:
                # fill the gap
                for fill_i in range(i_step, self.episode_limit):
                    self.episode_agent_pos.append(self.env.agent.position.tolist())
                    self.episode_agent_action.append(action.squeeze().tolist())
                    self.episode_done.append(done)
                break

            # evaluate data
            self.episode_agent_pos.append(self.env.agent.position.tolist())
            self.episode_agent_action.append(action.squeeze().tolist())
            self.episode_done.append(done)

        self.episode_reward = np.sum(reward_episode)
        self.episode_end_time = i_step

    def updateEpsilon(self):
        # print(self.epsilon_decay, self.step_count, self.epsilon)
        self.epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * np.exp(
            -1. * self.step_count / self.epsilon_decay)

    def update(self):
        # update epsilon
        self.step_count += 1
        self.updateEpsilon()

        # TODO check bsize_update
        bsize = min(self.experience_replay.current_size, self.bsize_update)
        batch = self.experience_replay.sample(bsize)

        action, reward, done = batch['a'], batch['r'], batch['done']  # (bsize, )
        state, state_next = batch['s'].to(DEVICE), batch['s_next'].to(DEVICE)  # (bsize, obs_dim)


        # process q_target, and q_eval
        q_eval, q_target = self.model(state), self.target_model(state_next)
        q_eval = torch.gather(q_eval, dim=-1, index=action.to(DEVICE)).squeeze(-1)
        q_target = q_target.max(dim=-1)[0].detach()

        # update model
        target = reward.to(DEVICE) + self.gamma * q_target * (1 - done).to(DEVICE)
        td_error = q_eval - target
        self.loss = (td_error ** 2).mean()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters())#, self.grad_norm_clip)

        return self.loss.item()

    def saveLog(self, ):
        # TODO check saveLOG
        # process name
        # time_string = '_' + time.strftime("%m%d") if self.save_time else ''
        file_name = self.file_name + '_' + str(self.ep_count)
        print("save {} log".format(file_name))

        # save train_log
        log_path = '../output/reinforceModel/log/' + file_name + '_train_log.txt'
        self.train_log.writeToFile(log_path)
        self.last_log_path = log_path

        #save evaluate_log
        log_path = '../output/reinforceModel/log/' + file_name + '_evaluate_log.txt'
        self.evaluate_log.writeToFile(log_path)
        # self.last_log_path = log_path

        # save model
        weight_path = '../output/reinforceModel/modelWeight/' + file_name + ".pth"
        torch.save({
            'model': self.model.state_dict(),
        }, weight_path)

        # saveEvaluatePng

        list_maps = ['agent_centric_map', 'source_centric_map', 'water_centric_map', 'agent_centric_epsilon_map',
                     'source_centric_epsilon_map', 'water_centric_epsilon_map']

        shown_map_path = '../output/reinforceModel/png/'
        for map_name in list_maps:
            saveShownMap(map_name, self.evaluate_dic_item[map_name], shown_map_path, file_name+'_')

    def showEpsilon(self):
        def calculateEpsilon(i):
            return self.epsilon_low + (self.epsilon_high - self.epsilon_low) * np.exp(
                -1. * i / self.epsilon_decay)

        plt.plot([calculateEpsilon(i * self.episode_limit) for i in range(self.n_ep_training)])
        plt.show()
