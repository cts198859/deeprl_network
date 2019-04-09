import numpy as np
import tensorflow as tf
from agents.utils import *


class Policy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def prepare_loss(self, v_coef, e_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * e_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        summaries = []
        summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
        summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
        summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
        summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
        summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
        summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
        self.summary = tf.summary.merge(summaries)

    def _build_actor_head(self, h, agent_name=None):
        name = 'pi'
        if agent_name is not None:
            name += '_' + str(agent_name)
        pi = fc(h, name, self.n_a, act=tf.nn.softmax)
        return tf.squeeze(pi)

    def _build_critic_head(self, h, na, agent_name=None):
        name = 'v'
        if agent_name is not None:
            name += '_' + str(agent_name)
        n_n = na.shape[-1]
        na_sparse = tf.one_hot(na, self.n_a, axis=-1)
        na_sparse = tf.reshape(na_sparse, [-1, self.n_a*n_n])
        h = tf.concat([h, na_sparse], 1)
        v = fc(h, name, 1, act=lambda x: x)
        return tf.squeeze(v)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s]) # forward 1-step
        self.naction_fw = tf.placeholder(tf.int32, [1, n_n])
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s]) # backward n-step
        self.naction_bw = tf.placeholder(tf.int32, [n_step, n_n])
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [n_lstm * 2])
        with tf.variable_scope(self.name):
            self.pi_fw, self.v_fw, self.new_states = self._build_net('forward')
        with tf.variable_scope(self.name, reuse=True):
            self.pi, self.v, _ = self._build_net('backward')
        self._reset()

    def backward(self, sess, obs, nactions, acts, dones, Rs, Advs, cur_lr,
                 summary_writer=None, global_step=None):
        summary, _ = sess.run([self.summary, self._train],
                              {self.ob_bw: obs,
                               self.naction_bw: nactions,
                               self.done_bw: dones,
                               self.states: self.states_bw,
                               self.A: acts,
                               self.ADV: Advs,
                               self.R: Rs,
                               self.lr: cur_lr})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)

    def forward(self, sess, ob, done, naction=None, out_type='p'):
        # update state only when p is called
        ins = {self.ob_fw: np.array([ob]),
               self.done_fw: np.array([done]),
               self.states: self.states_fw}
        if out_type.startswith('p'):
            outs = [self.pi_fw, self.new_states]
        else:
            outs = [self.v_fw]
            ins[self.naction_fw] = np.array([naction])
        out_values = sess.run(outs, ins)
        out_value = out_values[0]
        if out_type.startswith('p'):
            self.states_fw = out_values[-1]
        return out_value

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
            naction = self.naction_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
            naction = self.naction_bw
        h = fc(ob, 'fc', self.n_fc)
        h, new_states = lstm(h, done, self.states, 'lstm')
        pi = self._build_actor_head(h)
        v = self._build_critic_head(h, naction)
        return pi, v, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros(self.n_lstm * 2, dtype=np.float32)
        self.states_bw = np.zeros(self.n_lstm * 2, dtype=np.float32)


class NCMultiAgentPolicy(Policy):
    """ Inplemented as a centralized agent. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64):
        super().__init__(n_a, n_s, n_step, 'nc')
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask #n_agent x n_agent
        self.n_h = n_h
        self.ob_fw = tf.placeholder(tf.float32, [n_agent, 1, n_s]) # forward 1-step
        self.policy_fw = tf.placeholder(tf.float32, [n_agent, 1, n_a])
        self.action_fw = tf.placeholder(tf.int32, [n_agent, 1])
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_agent, n_step, n_s]) # backward n-step
        self.policy_bw = tf.placeholder(tf.float32, [n_agent, n_step, n_a])
        self.action_bw = tf.placeholder(tf.int32, [n_agent, n_step])
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [n_agent, n_h * 2])

        with tf.variable_scope(self.name):
            self.pi_fw, self.v_fw, self.new_states = self._build_net('forward')
        with tf.variable_scope(self.name, reuse=True):
            self.pi, self.v, _ = self._build_net('backward')
        self._reset()

    def backward(self, sess, obs, policies, acts, dones, Rs, Advs, cur_lr,
                 summary_writer=None, global_step=None):
        summary, _ = sess.run([self.summary, self._train],
                              {self.ob_bw: obs,
                               self.policy_bw: policies,
                               self.action_bw: acts,
                               self.done_bw: dones,
                               self.states: self.states_bw,
                               self.ADV: Advs,
                               self.R: Rs,
                               self.lr: cur_lr})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)

    def forward(self, sess, ob, done, policy, action=None, out_type='p'):
        # update state only when p is called
        ins = {self.ob_fw: np.expand_dims(ob, axis=1),
               self.done_fw: np.expand_dims(done, axis=1),
               self.policy_fw: np.expand_dims(policy, axis=1),
               self.states: self.states_fw}
        if out_type.startswith('p'):
            outs = [self.pi_fw, self.new_states]
        else:
            outs = [self.v_fw]
            ins[self.action_fw] = np.expand_dims(action, axis=1)
        out_values = sess.run(outs, ins)
        out_value = out_values[0]
        if out_type.startswith('p'):
            self.states_fw = out_values[-1]
        return out_value

    def prepare_loss(self, v_coef, e_coef, max_grad_norm, alpha, epsilon):
        self.ADV = tf.placeholder(tf.float32, [self.n_agent, self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_agent, self.n_step])
        A_sparse = tf.one_hot(self.action_bw, self.n_a)
        # all losses are averaged over steps but summed over agents
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=-1)
        entropy_loss = -tf.reduce_sum(tf.reduce_mean(entropy, axis=-1)) * e_coef
        policy_loss = -tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=-1) * self.ADV, axis=-1))
        value_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.R - self.v), axis=-1)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        summaries = []
        summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
        summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
        summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
        summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
        summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
        summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
        self.summary = tf.summary.merge(summaries)

    def _build_net(self, in_type):
        if in_type == 'forward':
            ob = self.ob_fw
            policy = self.policy_fw
            action = self.action_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            policy = self.policy_bw
            action = self.action_bw
            done = self.done_bw
        h, new_states = lstm_comm(ob, policy, done, self.neighbor_mask, self.states, 'lstm_comm')
        pi_ls = []
        v_ls = []
        for i in range(self.n_agent):
            h_i = h[i] # Txn_h
            mask_i = self.neighbor_mask[i]
            naction_i = tf.transpose(tf.boolean_mask(action, mask_i)) # Txn_n
            pi = self._build_actor_head(h_i, agent_name='%d' % i)
            v = self._build_critic_head(h_i, naction_i, agent_name='%d' % i)
            pi_ls.append(pi)
            v_ls.append(v)
        return tf.concat(pi_ls, axis=0), tf.concat(v_ls, axis=0), new_states

    def _reset(self):
        self.states_fw = np.zeros((self.n_agent, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((self.n_agent, self.n_lstm * 2), dtype=np.float32)

