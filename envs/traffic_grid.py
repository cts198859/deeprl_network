"""
Particular class of large traffic grid
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from envs.env import Phase, TrafficSimulator
from large_grid.data.build_file import gen_rou_file

sns.set_color_codes()


STATE_MEAN_MASKS = {'in_car': False, 'in_speed': True, 'out_car': False}
STATE_NAMES = ['in_car', 'in_speed', 'out_car']
PHASE_NUM = 2
# map from ild order (alphabeta) to signal order (clockwise from north)
STATE_PHASE_MAP = {'nt1': [2, 3, 1, 0], 'nt2': [2, 3, 1, 0],
                   'nt3': [2, 3, 1, 0], 'nt4': [2, 3, 1, 0],
                   'nt5': [2, 1, 0, 3], 'nt6': [3, 2, 0, 1],
                   'nt7': [0, 2, 3, 1], 'nt8': [0, 2, 3, 1],
                   'nt9': [1, 0, 2, 3], 'nt10': [1, 0, 2, 3],
                   'nt11': [3, 1, 0, 2], 'nt12': [3, 1, 0, 2],
                   'nt13': [3, 1, 0, 2], 'nt14': [3, 1, 0, 2],
                   'nt15': [1, 2, 3, 0], 'nt16': [3, 2, 1, 0],
                   'nt17': [2, 3, 1, 0], 'nt18': [2, 3, 1, 0],
                   'nt19': [2, 3, 1, 0], 'nt20': [1, 2, 3, 0],
                   'nt21': [0, 3, 2, 1], 'nt22': [0, 2, 3, 1],
                   'nt23': [0, 2, 3, 1], 'nt24': [0, 2, 3, 1],
                   'nt25': [1, 0, 2, 3]}


class LargeGridPhase(Phase):
    def __init__(self):
        two_phase = []
        phase = {'green': 'GGgsrrGGgsrr', 'yellow': 'yyysrryyysrr'}
        two_phase.append(phase)
        phase = {'green': 'srrGGgsrrGGg', 'yellow': 'srryyysrryyy'}
        two_phase.append(phase)
        self.phases = {2: two_phase}


class LargeGridController:
    def __init__(self, nodes):
        self.name = 'naive'
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node in zip(obs, self.nodes):
            actions.append(self.greedy(ob, node))
        return actions

    def greedy(self, ob, node):
        # hard code the mapping from state to number of cars
        phase = STATE_PHASE_MAP[node]
        in_cars = np.zeros_like(phase)
        for i, x in zip(phase, ob[:len(phase)]):
            in_cars[i] = x
        if (in_cars[0] + in_cars[2]) > (in_cars[1] + in_cars[3]):
            return 0
        return 1


class LargeGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.num_ext_car_hourly = config.getint('num_ext_car_per_hour')
        self.num_int_car_hourly = config.getint('num_int_car_per_hour')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_cross_action_num(self, node):
        return PHASE_NUM

    def _init_large_neighbor_map(self):
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
        return neighbor_map

    def _init_large_distance_map(self):
        distance_map = {}
        # corner nodes
        distance_map['nt1'] = {'nt3':2, 'nt7':2, 'nt11':2,
                               'nt4':3, 'nt8':3, 'nt12':3, 'nt16':3,
                               'nt5':4, 'nt9':4, 'nt13':4, 'nt17':4, 'nt21':4,
                               'nt10':5, 'nt14':5, 'nt18':5, 'nt22':5}
        distance_map['nt5'] = {'nt3':2, 'nt9':2, 'nt15':2,
                               'nt2':3, 'nt8':3, 'nt14':3, 'nt20':3,
                               'nt1':4, 'nt7':4, 'nt13':4, 'nt19':4, 'nt25':4,
                               'nt6':5, 'nt12':5, 'nt18':5, 'nt24':5}
        distance_map['nt21'] = {'nt11':2, 'nt17':2, 'nt23':2,
                                'nt6':3, 'nt12':3, 'nt18':3, 'nt24':3,
                                'nt1':4, 'nt7':4, 'nt13':4, 'nt19':4, 'nt25':4,
                                'nt2':5, 'nt8':5, 'nt14':5, 'nt20':5}
        distance_map['nt25'] = {'nt15':2, 'nt19':2, 'nt23':2,
                                'nt10':3, 'nt14':3, 'nt18':3, 'nt22':3,
                                'nt5':4, 'nt9':4, 'nt13':4, 'nt17':4, 'nt21':4,
                                'nt4':5, 'nt8':5, 'nt12':5, 'nt16':5}
        # edge nodes
        distance_map['nt2'] = {'nt4':2, 'nt6':2, 'nt8':2, 'nt12':2,
                               'nt5':3, 'nt9':3, 'nt11':3, 'nt13':3, 'nt17':3,
                               'nt10':4, 'nt14':4, 'nt16':4, 'nt18':4, 'nt22':4,
                               'nt15':5, 'nt19':5, 'nt21':5, 'nt23':5}
        distance_map['nt3'] = {'nt1':2, 'nt5':2, 'nt7':2, 'nt9':2, 'nt13':2,
                               'nt6':3, 'nt10':3, 'nt12':3, 'nt14':3, 'nt18':3,
                               'nt11':4, 'nt15':4, 'nt17':4, 'nt19':4, 'nt23':4,
                               'nt16':5, 'nt20':5, 'nt22':5, 'nt24':5}
        distance_map['nt4'] = {'nt2':2, 'nt8':2, 'nt10':2, 'nt14':2,
                               'nt1':3, 'nt7':3, 'nt13':3, 'nt15':3, 'nt19':3,
                               'nt6':4, 'nt12':4, 'nt18':4, 'nt20':4, 'nt24':4,
                               'nt11':5, 'nt17':5, 'nt23':5, 'nt25':5}
        distance_map['nt22'] = {'nt12':2, 'nt16':2, 'nt18':2, 'nt24':2,
                                'nt7':3, 'nt11':3, 'nt13':3, 'nt19':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt8':4, 'nt14':4, 'nt20':4,
                                'nt1':5, 'nt3':5, 'nt9':5, 'nt15':5}
        distance_map['nt23'] = {'nt13':2, 'nt17':2, 'nt19':2, 'nt21':2, 'nt25':2,
                                'nt8':3, 'nt12':3, 'nt14':3, 'nt16':3, 'nt20':3,
                                'nt3':4, 'nt7':4, 'nt9':4, 'nt11':4, 'nt15':4,
                                'nt2':5, 'nt4':5, 'nt6':5, 'nt10':5}
        distance_map['nt24'] = {'nt14':2, 'nt18':2, 'nt20':2, 'nt22':2,
                                'nt9':3, 'nt13':3, 'nt15':3, 'nt17':3, 'nt21':3,
                                'nt4':4, 'nt8':4, 'nt10':4, 'nt12':4, 'nt16':4,
                                'nt3':5, 'nt5':5, 'nt7':5, 'nt11':5}
        distance_map['nt10'] = {'nt4':2, 'nt8':2, 'nt14':2, 'nt20':2,
                                'nt3':3, 'nt7':3, 'nt13':3, 'nt19':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt12':4, 'nt18':4, 'nt24':4,
                                'nt1':5, 'nt11':5, 'nt17':5, 'nt23':5}
        distance_map['nt15'] = {'nt5':2, 'nt9':2, 'nt13':2, 'nt19':2, 'nt25':2,
                                'nt4':3, 'nt8':3, 'nt12':3, 'nt18':3, 'nt24':3,
                                'nt3':4, 'nt7':4, 'nt11':4, 'nt13':4, 'nt23':4,
                                'nt2':5, 'nt6':5, 'nt16':5, 'nt22':5}
        distance_map['nt20'] = {'nt10':2, 'nt14':2, 'nt18':2, 'nt24':2,
                                'nt5':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt23':3,
                                'nt4':4, 'nt8':4, 'nt12':4, 'nt16':4, 'nt22':4,
                                'nt3':5, 'nt7':5, 'nt11':5, 'nt21':5}
        distance_map['nt6'] = {'nt2':2, 'nt8':2, 'nt12':2, 'nt16':2,
                               'nt3':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt21':3,
                               'nt4':4, 'nt10':4, 'nt14':4, 'nt18':4, 'nt22':4,
                               'nt5':5, 'nt15':5, 'nt19':5, 'nt23':5}
        distance_map['nt11'] = {'nt1':2, 'nt7':2, 'nt13':2, 'nt17':2, 'nt21':2,
                                'nt2':3, 'nt8':3, 'nt14':3, 'nt18':3, 'nt22':3,
                                'nt3':4, 'nt9':4, 'nt15':4, 'nt19':4, 'nt23':4,
                                'nt4':5, 'nt10':5, 'nt20':5, 'nt24':5}
        distance_map['nt16'] = {'nt2':2, 'nt8':2, 'nt12':2, 'nt16':2,
                                'nt3':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt21':3,
                                'nt4':4, 'nt10':4, 'nt14':4, 'nt18':4, 'nt22':4,
                                'nt5':5, 'nt15':5, 'nt19':5, 'nt23':5}
        # internal nodes
        distance_map['nt7'] = {'nt1':2, 'nt3':2, 'nt9':2, 'nt11':2, 'nt13':2, 'nt17':2,
                               'nt4':3, 'nt10':3, 'nt14':3, 'nt16':3, 'nt18':3, 'nt22':3,
                               'nt5':4, 'nt15':4, 'nt19':4, 'nt21':4, 'nt23':4,
                               'nt20':5, 'nt24':5}
        distance_map['nt8'] = {'nt2':2, 'nt4':2, 'nt6':2, 'nt10':2, 'nt12':2, 'nt14':2, 'nt18':2,
                               'nt1':3, 'nt5':3, 'nt11':3, 'nt15':3, 'nt17':3, 'nt19':3, 'nt23':3,
                               'nt16':4, 'nt20':4, 'nt22':4, 'nt24':4,
                               'nt21':5, 'nt25':5}
        distance_map['nt9'] = {'nt3':2, 'nt5':2, 'nt7':2, 'nt13':2, 'nt15':2, 'nt19':2,
                               'nt2':3, 'nt6':3, 'nt12':3, 'nt18':3, 'nt20':3, 'nt24':3,
                               'nt1':4, 'nt11':4, 'nt17':4, 'nt23':4, 'nt25':4,
                               'nt16':5, 'nt22':5}
        distance_map['nt12'] = {'nt2':2, 'nt6':2, 'nt8':2, 'nt14':2, 'nt16':2, 'nt18':2, 'nt22':2,
                                'nt1':3, 'nt3':3, 'nt9':3, 'nt15':3, 'nt19':3, 'nt21':3, 'nt23':3,
                                'nt4':4, 'nt10':4, 'nt20':4, 'nt24':4,
                                'nt5':5, 'nt25':5}
        distance_map['nt13'] = {'nt3':2, 'nt7':2, 'nt9':2, 'nt11':2, 'nt15':2, 'nt17':2, 'nt19':2, 'nt23':2,
                                'nt2':3, 'nt4':3, 'nt6':3, 'nt10':3, 'nt16':3, 'nt20':3, 'nt22':3, 'nt24':3,
                                'nt1':4, 'nt5':4, 'nt21':4, 'nt25':4}
        distance_map['nt14'] = {'nt4':2, 'nt8':2, 'nt10':2, 'nt12':2, 'nt18':2, 'nt20':2, 'nt24':2,
                                'nt3':3, 'nt5':3, 'nt7':3, 'nt11':3, 'nt17':3, 'nt23':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt16':4, 'nt22':4,
                                'nt1':5, 'nt21':5}
        distance_map['nt17'] = {'nt7':2, 'nt11':2, 'nt13':2, 'nt19':2, 'nt21':2, 'nt23':2,
                                'nt2':3, 'nt6':3, 'nt8':3, 'nt14':3, 'nt20':3, 'nt24':3,
                                'nt1':4, 'nt3':4, 'nt9':4, 'nt15':4, 'nt25':4,
                                'nt4':5, 'nt10':5}
        distance_map['nt18'] = {'nt8':2, 'nt12':2, 'nt14':2, 'nt16':2, 'nt20':2, 'nt22':2, 'nt24':2,
                                'nt3':3, 'nt7':3, 'nt9':3, 'nt11':3, 'nt15':3, 'nt21':3, 'nt25':3,
                                'nt2':4, 'nt4':4, 'nt6':4, 'nt10':4,
                                'nt1':5, 'nt5':5}
        distance_map['nt19'] = {'nt9':2, 'nt13':2, 'nt15':2, 'nt17':2, 'nt23':2, 'nt25':2,
                                'nt4':3, 'nt8':3, 'nt10':3, 'nt12':3, 'nt16':3, 'nt22':3,
                                'nt3':4, 'nt5':4, 'nt7':4, 'nt11':4, 'nt21':4,
                                'nt2':5, 'nt6':5}
        return distance_map

    def _init_map(self):
        self.neighbor_map = self._init_large_neighbor_map()
        # for spatial discount
        self.distance_map = self._init_large_distance_map()
        self.phase_map = LargeGridPhase()
        self.state_names = STATE_NAMES
        self.state_mean_masks = STATE_MEAN_MASKS

    def _init_sim_config(self):
        return gen_rou_file(self.data_path,
                            self.num_ext_car_hourly,
                            self.num_int_car_hourly,
                            seed=self.seed,
                            thread=self.sim_thread)

    def plot_stat(self, rewards):
        data_set = {}
        data_set['car_num'] = np.array(self.car_num_stat)
        data_set['car_speed'] = np.array(self.car_speed_stat)
        data_set['reward'] = rewards
        for name, data in data_set.items():
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
    config.read('./config/config_test_large.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = LargeGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=False, record_stat=True)
    ob = env.reset()
    controller = LargeGridController(env.control_nodes)
    rewards = []
    while True:
        next_ob, reward, done, _ = env.step(controller.forward(ob))
        rewards += list(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
