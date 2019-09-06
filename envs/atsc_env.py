"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET

DEFAULT_PORT = 8000
SEC_IN_MS = 1000
VEH_LEN_M = 7.5 # effective vehicle length
QUEUE_MAX = 10


class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases
        self._init_phase_set()

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control # disabled
        self.ilds_in = [] # for state
        self.lanes_capacity = []
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0 # wave and wait should have the same dim
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_nodes()
        self.terminate()

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)
        # delete the current xml
        cmd = 'rm ' + trip_file
        subprocess.check_call(cmd, shell=True)

    def get_fingerprint(self):
        policies = []
        for node_name in self.node_names:
            policies.append(self.nodes[node_name].fingerprint)
        return policies

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=0):
        # have to terminate previous sim before calling reset
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        # initialize fingerprint
        self.update_fingerprint(self._init_policy())
        # next environment random condition should be different
        self.seed += 1
        return self._get_state()

    def step(self, action):
        self._set_phase(action, 'yellow', self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action, 'green', rest_interval_sec)
        self._simulate(rest_interval_sec)
        state = self._get_state()
        reward = self._measure_reward_step()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True
        global_reward = np.sum(reward)
        if self.is_record:
            action_r = ','.join(['%d' % a for a in action])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action': action_r,
                           'reward': global_reward}
            self.control_data.append(cur_control)

        # use original rewards in test
        if not self.train_mode:
            return state, reward, done, global_reward
        if (self.agent == 'greedy') or (self.coop_gamma < 0):
            reward = global_reward
        return state, reward, done, global_reward

    def terminate(self):
        self.sim.close()

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = pi

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        cur_phase = self.phase_map.get_phase(node.phase_id, action)
        if phase_type == 'green':
            return cur_phase
        prev_action = node.prev_action
        node.prev_action = action
        if (prev_action < 0) or (action == prev_action):
            return cur_phase
        prev_phase = self.phase_map.get_phase(node.phase_id, prev_action)
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase
        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)

    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_state(self):
        # hard code the state ordering as wave, wait, fp
        state = []
        # measure the most recent state
        self._measure_state_step()

        # get the appropriate state vectors
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave is required in state
            if self.agent == 'greedy':
                state.append(node.wave_state)
            else:
                cur_state = [node.wave_state]

                # include wave states of neighbors
                if self.agent.startswith('ia2c'):
                    for nnode_name in node.neighbor:
                        cur_state.append(self.nodes[nnode_name].wave_state)

                # include fingerprints of neighbors
                if self.agent == 'ia2c_fp':
                    for nnode_name in node.neighbor:
                        cur_state.append(self.nodes[nnode_name].fingerprint)

                # include wait state
                if 'wait' in self.state_names:
                    cur_state.append(node.wait_state)
                state.append(np.concatenate(cur_state))
        return state

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_agent = self.n_node
        # to simplify the sim, we assume all agents have the max action dim,
        # with tailing zeros during run time
        self.n_a_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            phase_num = self.phase_map.get_phase_num(phase_id)
            node.phase_id = phase_id
            node.n_a = phase_num
            self.n_a_ls.append(phase_num)

    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_nodes(self):
        nodes = {}
        tl_nodes = self.sim.trafficlight.getIDList()
        for node_name in self.node_names:
            if node_name not in tl_nodes:
                logging.error('node %s can not be found!' % node_name)
                exit(1)
            neighbor = self.neighbor_map[node_name]
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True)
            # controlled lanes: l:j,i_k
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            ilds_in = []
            lanes_cap = []
            for lane_name in lanes_in:
                if self.name == 'atsc_real_net':
                    cur_ilds_in = [lane_name]
                    if (node_name, lane_name) in self.extended_lanes:
                        cur_ilds_in += self.extended_lanes[(node_name, lane_name)]
                    ilds_in.append(cur_ilds_in)
                    cur_cap = 0
                    for ild_name in cur_ilds_in:
                        cur_cap += self.sim.lane.getLength(ild_name)
                    lanes_cap.append(cur_cap/float(VEH_LEN_M))
                else:
                    ilds_in.append(lane_name)
            nodes[node_name].ilds_in = ilds_in
            if self.name == 'atsc_real_net':
                nodes[node_name].lanes_capacity = lanes_cap
        self.nodes = nodes
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node_name in self.node_names:
            s += node_name + ':\n'
            node = self.nodes[node_name]
            s += '\tneigbor: %r\n' % node.neighbor
            s += '\tilds_in: %r\n' % node.ilds_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_policy(self):
        return [np.ones(self.n_a_ls[i]) / self.n_a_ls[i] for i in range(self.n_agent)]

    def _init_sim(self, seed, gui=False):
        sumocfg_file = self._init_sim_config(seed)
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '600'] # long teleport for safety
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        subprocess.Popen(command)
        # wait 1s to establish the traci server
        time.sleep(1)
        self.sim = traci.connect(port=self.port)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            node.num_state = len(node.ilds_in)
        for node_name in self.node_names:
            node = self.nodes[node_name]
            num_wave = node.num_state
            num_wait = 0 if 'wait' not in self.state_names else node.num_state
            if not self.agent.startswith('ma2c'):
                for nnode_name in node.neighbor:
                    num_wave += self.nodes[nnode_name].num_state
            self.n_s_ls.append(num_wait + num_wave)

    def _measure_reward_step(self):
        rewards = []
        for node_name in self.node_names:
            queues = []
            waits = []
            for ild in self.nodes[node_name].ilds_in:
                if self.obj in ['queue', 'hybrid']:
                    if self.name == 'atsc_real_net':
                        cur_queue = self.sim.lane.getLastStepHaltingNumber(ild[0])
                        cur_queue = min(cur_queue, QUEUE_MAX)
                    else:
                        cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)
                    queues.append(cur_queue)
                if self.obj in ['wait', 'hybrid']:
                    max_pos = 0
                    car_wait = 0
                    if self.name == 'atsc_real_net':
                        cur_cars = self.sim.lane.getLastStepVehicleIDs(ild[0])
                    else:
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                    for vid in cur_cars:
                        car_pos = self.sim.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.sim.vehicle.getWaitingTime(vid)
                    waits.append(car_wait)
            queue = np.sum(np.array(queues)) if len(queues) else 0
            wait = np.sum(np.array(waits)) if len(waits) else 0
            if self.obj == 'queue':
                reward = - queue
            elif self.obj == 'wait':
                reward = - wait
            else:
                reward = - queue - self.coef_wait * wait
            rewards.append(reward)
        return np.array(rewards)

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for k, ild in enumerate(node.ilds_in):
                        if self.name == 'atsc_real_net':
                            cur_wave = 0
                            for ild_seg in ild:
                                cur_wave += self.sim.lane.getLastStepVehicleNumber(ild_seg)
                            cur_wave /= node.lanes_capacity[k]
                            # cur_wave = min(1.5, cur_wave / QUEUE_MAX)
                        else:
                            cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
                        cur_state.append(cur_wave)
                    cur_state = np.array(cur_state)
                elif state_name == 'wait':
                    cur_state = []
                    for ild in node.ilds_in:
                        max_pos = 0
                        car_wait = 0
                        if self.name == 'atsc_real_net':
                            cur_cars = self.sim.lane.getLastStepVehicleIDs(ild[0])
                        else:
                            cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                        for vid in cur_cars:
                            car_pos = self.sim.vehicle.getLanePosition(vid)
                            if car_pos > max_pos:
                                max_pos = car_pos
                                car_wait = self.sim.vehicle.getWaitingTime(vid)
                        cur_state.append(car_wait)
                    cur_state = np.array(cur_state)
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                if state_name == 'wave':
                    node.wave_state = norm_cur_state
                else:
                    node.wait_state = norm_cur_state

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []
        for node_name in self.node_names:
            for ild in self.nodes[node_name].ilds_in:
                if self.name == 'atsc_real_net':
                    cur_queue = 0
                    for ild_seg in ild:
                        cur_queue += self.sim.lane.getLastStepHaltingNumber(ild_seg)
                else:
                    cur_queue = self.sim.lane.getLastStepHaltingNumber(ild)
                queues.append(cur_queue)
        queues = np.array(queues)
        avg_queue = np.mean(queues)
        std_queue = np.std(queues)
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}
        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0

    def _set_phase(self, action, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action)):
            phase = self._get_node_phase(a, node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)

    def _simulate(self, num_step):
        # reward = np.zeros(len(self.control_node_names))
        for _ in range(num_step):
            self.sim.simulationStep()
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()
