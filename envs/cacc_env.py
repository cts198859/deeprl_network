import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes()

HG_SCALE = 20

class CACCEnv:
    def __init__(self, config):
        self._load_config(config)
        self.ovm = OVMCarFollowing(self.h_s, self.h_g, self.v_max)
        self.train_mode = True
        self.cur_episode = 0
        self.is_record = False
        self._init_space()

    def _constrain_speed(self, v, u):
        # apply constraints
        v_next = v + np.clip(u, self.u_min, self.u_max) * self.dt
        v_next = np.clip(v_next, 0, self.v_max)
        u_const = (v_next - v) / self.dt
        return v_next, u_const

    def _get_human_accel(self, i, h_g):
        v = self.vs_cur[i]
        h = self.hs_cur[i]
        if i:
            v_lead = self.vs_cur[i-1]
        else:
            v_lead = self.v0s[self.t]
        alpha = self.alphas[i]
        beta = self.betas[i]
        return self.ovm.get_accel(v, v_lead, h, alpha, beta, h_g)

    def _get_reward(self):
        v_state = np.array(self.vs_cur, copy=True)
        h_state = np.array(self.hs_cur, copy=True)
        u_state = np.array(self.us_cur, copy=True)
        # give large penalty for collision
        if np.min(h_state) < self.h_min:
            return -self.G * np.ones(self.n_agent)
        h_rewards = -(h_state - self.h_star) ** 2
        v_rewards = -self.a * (v_state - self.v_star) ** 2
        u_rewards = -self.b * (u_state) ** 2
        if self.train_mode:
            c_rewards = self.c * (np.minimum(h_state - self.h_s, 0)) ** 2
        else:
            c_rewards = 0
        return h_rewards + v_rewards + u_rewards + c_rewards

    def _get_veh_state(self, i_veh):
        v_state = np.clip((self.vs_cur[i_veh] - self.v_star) / 5, -2, 2)
        h_state = np.clip((self.hs_cur[i_veh] - self.h_star) / 10, -2, 2)
        u_state = self.us_cur[i_veh] / self.u_max
        return np.array([v_state, h_state, u_state])

    def _get_state(self):
        state = []
        for i in range(self.n_agent):
            cur_state = [self._get_veh_state(i)]
            if self.agent.startswith('ia2c'):
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur_state.append(self._get_veh_state(j))
            if self.agent == 'ia2c_fp':
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur_state.append(self.fp[j])
            state.append(np.concatenate(cur_state))
        return state

    def _log_data(self):
        hs = np.array(self.hs)
        vs = np.array(self.vs)
        us = np.array(self.us)
        df = pd.DataFrame()
        df['episode'] = np.ones(len(hs)) * self.cur_episode
        df['time_sec'] = np.arange(len(hs)) * self.dt
        df['step'] = np.arange(len(hs))
        df['reward'] = np.array(self.rewards)
        df['lead_headway_m'] = hs[:, 0]
        df['avg_headway_m'] = np.mean(hs[:, 1:], axis=1)
        df['std_headway_m'] = np.std(hs[:, 1:], axis=1)
        df['avg_speed_mps'] = np.mean(vs, axis=1)
        df['std_speed_mps'] = np.std(vs, axis=1)
        df['avg_accel_mps2'] = np.mean(us, axis=1)
        df['std_accel_mps2'] = np.std(us, axis=1)
        # for i in range(self.n_agent):
        #     df['headway_%d' % (i+1)] = hs[:, i]
        #     df['velocity_%d' % (i+1)] = vs[:, i]
        #     df['control_%d' % (i+1)] = us[:, i]
        self.data.append(df)

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.output_path = output_path
        if self.is_record:
            self.data = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction

    def output_data(self):
        if not self.is_record:
            return
        df = pd.concat(self.data)
        df.to_csv(self.output_path + ('%s_%s.csv' % (self.name, self.agent)))
        # self.plot_data(df, path)

    def plot_data(self, df, path):
        fig = plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        for i in [0, 2, 5, 7]:
            plt.plot(df.time.values, df['headway_%d' % (i+1)].values, linewidth=3,
                     label='veh #%d' % (i+1))
        plt.legend(fontsize=20, loc='best')
        plt.grid(True, which='both')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Headway [m]', fontsize=20)
        plt.subplot(2, 1, 2)
        for i in [0, 2, 5, 7]:
            plt.plot(df.time.values, df['velocity_%d' % (i+1)].values, linewidth=3,
                     label='veh #%d' % (i+1))
        # plt.legend(fontsize=15, loc='best')
        plt.grid(True, which='both')
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Velocity [m/s]', fontsize=20)
        plt.xlabel('Time [s]', fontsize=20)
        fig.tight_layout()
        plt.savefig(path + 'env_plot.pdf')
        plt.close()

    def reset(self, h0=None, v0=None, gui=False, test_ind=0):
        self.cur_episode += 1
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        np.random.seed(seed)
        self._init_common()
        self.seed += 1
        if self.name.startswith('catchup'):
            self._init_catchup()
        elif self.name.startswith('slowdown'):
            self._init_slowdown()
        self.hs_cur = self.hs[0]
        self.vs_cur = self.vs[0]
        if h0 is not None:
            self.hs_cur[0] = h0
        if v0 is not None:
            self.vs_cur[0] = v0
        self.us_cur = [0] * self.n_agent
        self.fp = np.zeros((self.n_agent, 2*self.n_a))
        self.us = [self.us_cur]
        self.rewards = [0]
        return self._get_state()

    def step(self, action=0):
        auto_hgs = self.h_g + action * HG_SCALE
        hs_next = []
        vs_next = []
        self.us_cur = []
        # update speed
        for i in range(self.n_agent):
            h_g = auto_hgs[i]
            u = self._get_human_accel(i, h_g)
            v_next, u_const = self._constrain_speed(self.vs_cur[i], u)
            self.us_cur.append(u_const)
            vs_next.append(v_next)
        # update headway
        for i in range(self.n_agent):
            if i == 0:
                v_lead = self.v0s[self.t]
                v_lead_next = self.v0s[self.t+1]
            else:
                v_lead = self.vs_cur[i-1]
                v_lead_next = vs_next[i-1]
            v = self.vs_cur[i]
            v_next = vs_next[i]
            hs_next.append(self.hs_cur[i] + 0.5*self.dt*(v_lead+v_lead_next-v-v_next))
        self.hs_cur = hs_next
        self.vs_cur = vs_next
        self.hs.append(self.hs_cur)
        self.vs.append(self.vs_cur)
        self.us.append(self.us_cur)
        self.t += 1
        reward = self._get_reward()
        global_reward = np.sum(reward)
        self.rewards.append(global_reward)
        done = False
        if reward[0] == -self.G:
            done = True
        if self.t == self.T:
            done = True
        if self.coop_gamma < 0:
            reward = global_reward
        if done and (self.is_record):
            self._log_data()
        return self._get_state(), reward, done, global_reward

    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def _init_space(self):
        self.n_agent = 8
        self.neighbor_mask = np.zeros((self.n_agent, self.n_agent))
        self.distance_mask = np.zeros((self.n_agent, self.n_agent))
        cur_distance = np.arange(self.n_agent)
        for i in range(self.n_agent):
            self.distance_mask[i] = cur_distance
            cur_distance = np.concatenate([[i+1], cur_distance[:-1]])
            if i >= 1:
                self.neighbor_mask[i,i-1] = 1
            if i <= self.n_agent-2:
                self.neighbor_mask[i,i+1] = 1
        self.n_a = 1
        n_s_ls = []
        for i in range(self.n_agent):
            if self.agent.startswith('ma2c'):
                num_n = 1
            else:
                num_n = 1 + np.sum(self.neighbor_mask[i])
            n_s_ls.append(num_n * 3)
        if self.agent.startswith('ma2c'):
            assert len(set(n_s_ls)) == 1
            self.n_s = n_s_ls[0]
        else:
            self.n_s_ls = n_s_ls

    def _init_catchup(self):
        # only the first vehicle has long headway
        self.hs = [[self.h_star*4] + [self.h_star] * 7]
        self.vs = [[self.v_star] * 8]
        self.v0s = [self.v_star] * (self.T+1)

    def _init_common(self):
        self.alphas = np.random.rand(self.n_agent) * 0.75
        self.betas = np.random.rand(self.n_agent) * 0.75
        self.t = 0

    def _init_slowdown(self):
        self.hs = [[self.h_star/3*4] * 8]
        self.vs = [[self.v_star*2] * 8]
        v0s_decel = list(np.arange(self.v_star*2, self.v_star-0.1, self.u_min/2))
        self.v0s = v0s_decel + [self.v_star] * (self.T+1-len(v0s_decel))

    def _load_config(self, config):
        self.T = config.getint('episode_length')
        self.dt = config.getfloat('delta_t')
        self.h_min = config.getfloat('headway_min')
        self.h_star = config.getfloat('headway_target')
        self.h_s = config.getfloat('headway_st')
        self.h_g = config.getfloat('headway_go')
        self.v_max = config.getfloat('speed_max')
        self.v_star = config.getfloat('speed_target')
        self.u_min = config.getfloat('accel_min')
        self.u_max = config.getfloat('accel_max')
        self.name = config.get('scenario')
        self.a = config.getfloat('reward_a')
        self.b = config.getfloat('reward_b')
        self.c = config.getfloat('reward_c')
        self.G = config.getfloat('penalty')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.seed = config.getint('seed')
        test_seeds = config.get('test_seeds')
        if test_seeds == 'default':
            test_seeds = list(range(2000, 2500, 10))
        else:
            test_seeds = [int(s) for s in test_seeds.split(',')]
        self.init_test_seeds(test_seeds)


class OVMCarFollowing:
    '''
    A OVM controller for human-driven vehicles
    Attributes:
        h_st (float): stop headway
        h_go (float): full-speed headway
        v_max (float): max speed
    '''
    def __init__(self, h_st, h_go, v_max):
        """Initialization."""
        self.h_st = h_st
        self.h_go = h_go
        self.v_max = v_max

    def get_accel(self, v, v_lead, h, alpha, beta, h_go=-1):
        """
        Get target acceleration using OVM controller.

        Args:
            v (float): current vehicle speed
            v_lead (float): leading vehicle speed
            h (float): current headway
            alpha, beta (float): human parameters
        Returns:
            accel (float): target acceleration
        """
        if h_go < 0:
            h_go = self.h_go
        if h <= self.h_st:
            vh = 0
        elif self.h_st < h < h_go:
            vh = self.v_max / 2 * (1 - np.cos(np.pi * (h-self.h_st) / (h_go-self.h_st)))
            # vh = self.v_max * ((d-h_st) / (h_go-h_st))
        else:
            vh = self.v_max
        # alpha is applied to both headway based V and leading speed based V.
        return alpha*(vh-v) + beta*(v_lead-v)

