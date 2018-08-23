import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect


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

    def _build_fc_net(self, h, n_fc, out_type):
        h = fc(h, out_type + '_fc', n_fc)
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
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
        summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
        summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
        self.summary = tf.summary.merge(summaries)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, n_fc=128, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net(n_fc, 'forward', 'pi')
            self.v_fw, v_state = self._build_net(n_fc, 'forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net(n_fc, 'backward', 'pi')
            self.v, _ = self._build_net(n_fc, 'backward', 'v')
        self._reset()

    def _build_net(self, n_fc, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h, new_states = lstm(ob, done, states, out_type + '_lstm')
        out_val = self._build_fc_net(h, n_fc, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        summary, _ = sess.run([self.summary, self._train],
                              {self.ob_bw:obs,
                               self.done_bw:dones,
                               self.states:self.states_bw,
                               self.A:acts,
                               self.ADV:Advs,
                               self.R:Rs,
                               self.lr:cur_lr,
                               self.entropy_coef:cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class FcPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, n_fc0=256, n_fc=128, name=None):
        super().__init__(n_a, n_s, n_step, 'fc', name)
        self.n_fc = n_fc0
        self.obs = tf.placeholder(tf.float32, [None, n_s])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net(n_fc, 'pi')
            self.v = self._build_net(n_fc, 'v')

    def _build_net(self, n_fc, out_type):
        h = fc(self.obs, out_type + '_fc0', self.n_fc)
        return self._build_fc_net(h, n_fc, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs:[ob]})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        summary, _ = sess.run([self.summary, self._train],
                              {self.obs:obs,
                               self.A:acts,
                               self.ADV:Advs,
                               self.R:Rs,
                               self.lr:cur_lr,
                               self.entropy_coef:cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)


class HybridPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_f, n_step, n_fc0=128, n_fc=128, n_lstm=64, name=None):
        Policy.__init__(self, n_a, n_s, n_step, 'hybrid', name)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_fc0 = n_fc0
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_f]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_f]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net(n_fc, 'forward', 'pi')
            self.v_fw, v_state = self._build_net(n_fc, 'forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net(n_fc, 'backward', 'pi')
            self.v, _ = self._build_net(n_fc, 'backward', 'v')
        self._reset()

    def _build_net(self, n_fc, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0, new_states = lstm(ob[:, :self.n_s], done, states, out_type + '_lstm')
        h1 = fc(ob[:, self.n_s:], out_type + '_fc0', self.n_fc0)
        h = tf.concat([h0, h1], 1)
        out_val = self._build_fc_net(h, n_fc, out_type)
        return out_val, new_states
