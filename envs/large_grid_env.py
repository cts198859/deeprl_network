"""
ATSC scenario: large traffic grid
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.large_grid_data.build_file import gen_rou_file

sns.set_color_codes()


STATE_NAMES = ['wave']
PHASE_NUM = 5


class LargeGridPhase(PhaseMap):
    def __init__(self):
        phases = ['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr',
                  'rrrGGGrrrrrr', 'rrrrrrrrrGGG']
        self.phases = {PHASE_NUM: PhaseSet(phases)}


class LargeGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))


class LargeGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1')
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return PHASE_NUM

    def _init_neighbor_map(self):
        neighbor_map = {}
        # corner nodes
        neighbor_map['nt1'] = ['nt6', 'nt2']
        neighbor_map['nt5'] = ['nt10', 'nt4']
        neighbor_map['nt21'] = ['nt22', 'nt16']
        neighbor_map['nt25'] = ['nt20', 'nt24']
        # edge nodes
        neighbor_map['nt2'] = ['nt7', 'nt3', 'nt1']
        neighbor_map['nt3'] = ['nt8', 'nt4', 'nt2']
        neighbor_map['nt4'] = ['nt9', 'nt5', 'nt3']
        neighbor_map['nt22'] = ['nt23', 'nt17', 'nt21']
        neighbor_map['nt23'] = ['nt24', 'nt18', 'nt22']
        neighbor_map['nt24'] = ['nt25', 'nt19', 'nt23']
        neighbor_map['nt10'] = ['nt15', 'nt5', 'nt9']
        neighbor_map['nt15'] = ['nt20', 'nt10', 'nt14']
        neighbor_map['nt20'] = ['nt25', 'nt15', 'nt19']
        neighbor_map['nt6'] = ['nt11', 'nt7', 'nt1']
        neighbor_map['nt11'] = ['nt16', 'nt12', 'nt6']
        neighbor_map['nt16'] = ['nt21', 'nt17', 'nt11']
        # internal nodes
        for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
            n_node = 'nt' + str(i + 5)
            s_node = 'nt' + str(i - 5)
            w_node = 'nt' + str(i - 1)
            e_node = 'nt' + str(i + 1)
            cur_node = 'nt' + str(i)
            neighbor_map[cur_node] = [n_node, e_node, s_node, w_node]
        self.neighbor_map = neighbor_map
        self.neighbor_mask = np.zeros((self.n_node, self.n_node))
        for i in range(self.n_node):
            for nnode in neighbor_map['nt%d' % (i+1)]:
                ni = self.node_names.index(nnode)
                self.neighbor_mask[i, ni] = 1
        logging.info('neighbor mask:\n %r' % self.neighbor_mask)

    def _init_distance_map(self):
        block0 = np.array([[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]])
        block1 = block0 + 1
        block2 = block0 + 2
        block3 = block0 + 3
        block4 = block0 + 4
        row0 = np.hstack([block0, block1, block2, block3, block4])
        row1 = np.hstack([block1, block0, block1, block2, block3])
        row2 = np.hstack([block2, block1, block0, block1, block2])
        row3 = np.hstack([block3, block2, block1, block0, block1])
        row4 = np.hstack([block4, block3, block2, block1, block0])
        self.distance_mask = np.vstack([row0, row1, row2, row3, row4]) 

    def _init_map(self):
        self.node_names = ['nt%d' % i for i in range(1, 26)]
        self.n_node = 25
        self._init_neighbor_map()
        # for spatial discount
        self._init_distance_map()
        self.max_distance = 8
        self.phase_map = LargeGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        return gen_rou_file(self.data_path,
                            self.peak_flow1,
                            self.peak_flow2,
                            self.init_density,
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
    config.read('./config/config_greedy.ini')
    base_dir = './greedy/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = LargeGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(1)
    controller = LargeGridController(env.node_names)
    rewards = []
    for i in range(env.test_num):
        ob = env.reset(test_ind=i)
        while True:
            next_ob, _, done, reward = env.step(controller.forward(ob))
            rewards.append(reward)
            if done:
                break
            ob = next_ob
        env.terminate()
        time.sleep(2)
        env.collect_tripinfo()
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.output_data()
