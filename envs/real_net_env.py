"""
ATSC scenario: Monaco traffic network
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from collections import deque
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.real_net_data.build_file import gen_rou_file

sns.set_color_codes()

STATE_NAMES = ['wave']
# node: (phase key, neighbor list)
NODES = {'10026': ('6.0', ['9431', '9561', 'cluster_9563_9597', '9531']), 
         '8794': ('4.0', ['cluster_8985_9609', '9837', '9058', 'cluster_9563_9597']),
         '8940': ('2.1', ['9007', '9429']),
         '8996': ('2.2', []),
         '9007': ('2.3', ['9309', '8940']),
         '9058': ('4.0', ['cluster_8985_9609', '8794', 'joinedS_0']),
         '9153': ('2.0', ['9643']),
         '9309': ('4.0', ['9466', '9007', 'cluster_9043_9052']),
         '9413': ('2.3', ['9721', '9837']),
         '9429': ('5.0', ['cluster_9043_9052', '8940']),
         '9431': ('2.4', ['9721', '9884', '9561', '10026']),
         '9433': ('2.5', []),
         '9466': ('4.0', ['9309', 'joinedS_0']),
         '9480': ('2.3', []),
         '9531': ('2.6', ['joinedS_1']),
         '9561': ('4.0', ['cluster_9389_9689', '10026']),
         '9643': ('2.3', ['9153']),
         '9713': ('3.0', ['9721']),
         '9721': ('6.0', ['9431', '9713', '9413']),
         '9837': ('3.1', ['9413', '8794', 'cluster_8985_9609']),
         '9884': ('2.7', ['9713', 'cluster_9389_9689']),
         'cluster_8751_9630': ('4.0', []),
         'cluster_8985_9609': ('4.0', ['9837', '8794', '9058']),
         'cluster_9043_9052': ('4.1', ['cluster_9563_9597', '10026', 'joinedS_1']),
         'cluster_9389_9689': ('4.0', ['cluster_8751_9630', '9884', '9561', '8996']),
         'cluster_9563_9597': ('4.2', ['10026', '8794', 'joinedS_0', 'cluster_9043_9052']),
         'joinedS_0': ('6.1', ['9058', 'cluster_9563_9597', '9466']),
         'joinedS_1': ('3.2', ['9531', '9429'])}

PHASES = {'4.0': ['GGgrrrGGgrrr', 'rrrGGgrrrGGg', 'rrGrrrrrGrrr', 'rrrrrGrrrrrG'],
          '4.1': ['GGgrrGGGrrr', 'rrGrrrrrrrr', 'rrrGgrrrGGg', 'rrrrGrrrrrG'],
          '4.2': ['GGGGrrrrrrrr', 'GGggrrGGggrr', 'rrrGGGGrrrrr', 'grrGGggrrGGg'],
          '2.0': ['GGrrr', 'ggGGG'],
          '2.1': ['GGGrrr', 'rrGGGg'],
          '2.2': ['Grr', 'gGG'],
          '2.3': ['GGGgrr', 'GrrrGG'],
          '2.4': ['GGGGrr', 'rrrrGG'],
          '2.5': ['Gg', 'rG'],
          '2.6': ['GGGg', 'rrrG'],
          '2.7': ['GGg', 'rrG'],
          '3.0': ['GGgrrrGGg', 'rrGrrrrrG', 'rrrGGGGrr'],
          '3.1': ['GgrrGG', 'rGrrrr', 'rrGGGr'],
          '3.2': ['GGGGrrrGG', 'rrrrGGGGr', 'GGGGrrGGr'],
          '5.0': ['GGGGgrrrrGGGggrrrr', 'grrrGrrrrgrrGGrrrr', 'GGGGGrrrrrrrrrrrrr',
                  'rrrrrrrrrGGGGGrrrr', 'rrrrrGGggrrrrrggGg'],
          '6.0': ['GGGgrrrGGGgrrr', 'rrrGrrrrrrGrrr', 'GGGGrrrrrrrrrr', 'rrrrrrrrrrGGGG',
                  'rrrrGGgrrrrGGg', 'rrrrrrGrrrrrrG'],
          '6.1': ['GGgrrGGGrrrGGGgrrrGGGg', 'rrGrrrrrrrrrrrGrrrrrrG', 'GGGrrrrrGGgrrrrGGgrrrr',
                  'GGGrrrrrrrGrrrrrrGrrrr', 'rrrGGGrrrrrrrrrrrrGGGG', 'rrrGGGrrrrrGGGgrrrGGGg']}

EXTENDED_LANES = {('9431', '10099#3_1'): ['10099#1_1', '10099#2_1'],
                  ('10026', '-10046#0_1'): ['-10046#1_1'],
                  ('10026', '-10089#4_1'): ['10031#1_1'],
                  ('10026', '-10089#4_2'): ['10031#1_2'],
                  ('8940', '10065#1_1'): ['10065#0_1'],
                  ('9007', '-10065#0_1'): ['-10065#1_1'],
                  ('9429', '10064#3_1'): ['gneE12_0'],
                  ('9429', '10064#3_2'): ['gneE12_1'],
                  ('9431', '10099#3_1'): ['10099#1_1', '10099#2_1', 'gneE14_0'],
                  ('9433', '10052#5_1'): ['10052#4_1'],
                  ('9433', '10180#3_1'): ['10180#1_1'],
                  ('9466', '-10067#0_1'): ['-10067#1_1', '-10117#0_1'],
                  ('9480', '10183#13_1'): ['10183#12_1'],
                  ('9480', '-10183#14_1'): ['-10183#16_1'],
                  ('9531', '10077_1'): ['10116_1'],
                  ('9561', '10046#1_1'): ['10046#0_1'],
                  ('9643', '-10178_1'): ['-10179_1'],
                  ('9643', '-10051#1_1'): ['-10051#2_1'],
                  ('9713', '-10094#2_1'): ['-10094#3_1'],
                  ('9713', '10094#1_1'): ['10094#0_1', '10097#2_1'],
                  ('9721', '10094#3_1'): ['10094#2_1'],
                  ('cluster_8751_9630', '-10078#2_1'): ['-10078#3_1', '10085_1'],
                  ('cluster_9043_9052', '-10090#0_1'): ['-10090#1_1', '10080#2_1'],
                  ('cluster_9043_9052', '-10090#0_2'): ['-10090#1_2'],
                  ('cluster_9389_9689', '-10046#5_1'): ['10083#1_1', '-10083#2_1'],
                  ('cluster_9563_9597', '10090#1_1'): ['10090#0_1'],
                  ('cluster_9563_9597', '10090#1_2'): ['10090#0_2'],
                  ('cluster_9563_9597', '10108#5_1'): ['gneE8_0'],
                  ('cluster_9563_9597', '10108#5_2'): ['gneE8_1'],
                  ('joinedS_0', 'gneE7_0'): ['-10108#5_1'],
                  ('joinedS_0', 'gneE7_1'): ['-10108#5_2'],
                  ('joinedS_1', '10181#2_1'): ['10181#1_1']
                  }

class RealNetPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)


class RealNetController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set()
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))


class RealNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.flow_rate = config.getint('flow_rate')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _bfs(self, i):
        d = 0
        self.distance_mask[i, i] = d
        visited = [False]*self.n_node
        que = deque([i])
        visited[i] = True
        while que:
            d += 1
            for _ in range(len(que)):
                node_name = self.node_names[que.popleft()]
                for nnode in self.neighbor_map[node_name]:
                    ni = self.node_names.index(nnode)
                    if not visited[ni]:
                        self.distance_mask[i, ni] = d
                        visited[ni] = True
                        que.append(ni)
        return d

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _init_neighbor_map(self):
        self.neighbor_map = dict([(key, val[1]) for key, val in NODES.items()])
        self.neighbor_mask = np.zeros((self.n_node, self.n_node)).astype(int)
        for i, node_name in enumerate(self.node_names):
            for nnode in self.neighbor_map[node_name]:
                ni = self.node_names.index(nnode)
                self.neighbor_mask[i, ni] = 1
        logging.info('neighbor mask:\n %r' % self.neighbor_mask)

    def _init_distance_map(self):
        self.distance_mask = -np.ones((self.n_node, self.n_node)).astype(int)
        self.max_distance = 0
        for i in range(self.n_node):
            self.max_distance = max(self.max_distance, self._bfs(i))
        logging.info('distance mask:\n %r' % self.distance_mask)

    def _init_map(self):
        self.node_names = sorted(list(NODES.keys()))
        self.n_node = len(self.node_names)
        self._init_neighbor_map()
        self._init_distance_map()
        self.phase_map = RealNetPhase()
        self.phase_node_map = dict([(key, val[0]) for key, val in NODES.items()])
        self.state_names = STATE_NAMES
        self.extended_lanes = EXTENDED_LANES

    def _init_sim_config(self, seed):
        # comment out to call build_file.py
        return gen_rou_file(self.data_path,
                            self.flow_rate,
                            seed=seed,
                            thread=self.sim_thread)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_real.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = RealNetEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(1)
    # ob = env.reset(gui=True)
    controller = RealNetController(env.node_names, env.nodes)
    env.init_test_seeds(list(range(10000, 100001, 10000)))
    rewards = []
    for i in range(10):
        ob = env.reset(test_ind=i)
        global_rewards = []
        cur_step = 0
        while True:
            next_ob, reward, done, global_reward = env.step(controller.forward(ob))
            # for node_name, node_ob in zip(env.node_names, next_ob):
                # logging.info('%d, %s:%r\n' % (cur_step, node_name, node_ob))
            global_rewards.append(global_reward)
            rewards += list(reward)
            cur_step += 1
            if done:
                break
            ob = next_ob
        env.terminate()
        logging.info('step: %d, avg reward: %.2f' % (cur_step, np.mean(global_rewards)))
        time.sleep(1)
    env.plot_stat(np.array(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
