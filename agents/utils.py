import numpy as np
import torch
import torch.nn as nn

"""
initializers
"""
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

"""
layer helpers
"""
def batch_to_seq(x):
    n_step = x.shape[0].value
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)


def seq_to_batch(x):
    return torch.cat(x)


def run_rnn(layer, xs, dones, s):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        xs[ind] = h
    s = torch.cat([h, c], dim=1)
    return seq_to_batch(xs), torch.squeeze(s)


def one_hot(x, oh_dim, dim=-1):
    oh_shape = list(x.shape)
    oh_shape[dim] = oh_dim
    x_oh = torch.zeros(oh_shape)
    x_oh.scatter(dim, x, 1)
    return x_oh
    

def lstm_comm(xs, ps, dones, masks, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
              init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    n_s = xs.shape[-1]
    n_a = ps.shape[-1]
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    w_fp = []
    b_fp = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    n_in_hid = 3*n_h
    for i in range(n_agent):
        n_m = np.sum(masks[i])
        # n_in_hid = (n_m+1)*n_h
        with tf.variable_scope(scope + ('_%d' % i)):
            w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable("b_msg", [n_h],
                                         initializer=tf.constant_initializer(0.0)))
            w_ob.append(tf.get_variable("w_ob", [n_s*(n_m+1), n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            w_fp.append(tf.get_variable("w_fp", [n_a*n_m, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_fp.append(tf.get_variable("b_fp", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_in_hid, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        # x = batch_to_seq(tf.squeeze(x, axis=0))
        # p = batch_to_seq(tf.squeeze(p, axis=0))
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        # for i in range(n_agent):
            # hi = tf.expand_dims(h[i], axis=0)
            # hxi = fc(xi, 'mfc_s_%d' % i, n_h, act=tf.nn.tanh)
            # hpi = fc(pi, 'mfc_p_%d' % i, n_h, act=tf.nn.tanh)
            # si = tf.concat([hi, hxi, hpi], axis=1)
            # mi = fc(hi, 'mfc_%d' % i, n_h)
            # out_m.append(mi)
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
            pi = tf.expand_dims(tf.reshape(tf.boolean_mask(p, masks[i]), [-1]), axis=0)
            xi = tf.expand_dims(tf.reshape(tf.boolean_mask(x, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x[i], axis=0), xi], axis=1)
            hxi = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            hpi = tf.nn.relu(tf.matmul(pi, w_fp[i]) + b_fp[i])
            hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
            si = tf.concat([hxi, hpi, hmi], axis=1)
            # si = tf.concat([hxi, hmi], axis=1)
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_comm_hetero(xs, ps, dones, masks, s, n_s_ls, n_a_ls, scope, init_scale=DEFAULT_SCALE,
                     init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    w_fp = []
    b_fp = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    na_dim_ls = []
    ns_dim_ls = []
    for i in range(n_agent):
        n_s = n_s_ls[i]
        n_fp = 0
        na_dim = []
        ns_dim = []
        for j in np.where(masks[i])[0]:
            n_s += n_s_ls[j]
            n_fp += n_a_ls[j]
            na_dim.append(n_a_ls[j])
            ns_dim.append(n_s_ls[j])
        na_dim_ls.append(na_dim)
        ns_dim_ls.append(ns_dim)
        n_m = len(ns_dim)
        if n_m:
            n_in_hid = 3*n_h
        else:
            n_in_hid = n_h
        with tf.variable_scope(scope + ('_%d' % i)):
            w_ob.append(tf.get_variable("w_ob", [n_s, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            if n_m:
                w_fp.append(tf.get_variable("w_fp", [n_fp, n_h],
                                            initializer=init_method(init_scale, init_mode)))
                b_fp.append(tf.get_variable("b_fp", [n_h],
                                            initializer=tf.constant_initializer(0.0)))
                w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                             initializer=init_method(init_scale, init_mode)))
                b_msg.append(tf.get_variable("b_msg", [n_h],
                                             initializer=tf.constant_initializer(0.0)))
            else:
                w_fp.append(None)
                b_fp.append(None)
                w_msg.append(None)
                b_msg.append(None)
            wx_hid.append(tf.get_variable("wx_hid", [n_in_hid, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            n_m = len(ns_dim_ls[i])
            pi = []
            xi = [tf.slice(x, [i, 0], [1, n_s_ls[i]])]
            if n_m:
                mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
                raw_pi = tf.boolean_mask(p, masks[i]) # n_n*n_a
                raw_xi = tf.boolean_mask(x, masks[i])
                # find the valid information based on each agent's s, a dim
                for j in range(n_m):
                    pi.append(tf.slice(raw_pi, [j, 0], [1, na_dim_ls[i][j]]))
                    xi.append(tf.slice(raw_xi, [j, 0], [1, ns_dim_ls[i][j]]))
                xi = tf.concat(xi, axis=1)
            else:
                xi = xi[0]
            hxi = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            if n_m:
                hpi = tf.nn.relu(tf.matmul(tf.concat(pi, axis=1), w_fp[i]) + b_fp[i])
                hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
                si = tf.concat([hxi, hpi, hmi], axis=1)
            else:
                si = hxi
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_ic3(xs, dones, masks, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
             init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    n_s = xs.shape[-1]
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    for i in range(n_agent):
        n_m = np.sum(masks[i])
        with tf.variable_scope(scope + ('_%d' % i)):
            w_msg.append(tf.get_variable("w_msg", [n_h, n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable("b_msg", [n_h],
                                         initializer=tf.constant_initializer(0.0)))
            w_ob.append(tf.get_variable("w_ob", [n_s*(n_m+1), n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, done) in enumerate(zip(xs, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        out_h = []
        out_c = []
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            mi = tf.reduce_mean(tf.boolean_mask(out_m, masks[i]), axis=0, keepdims=True)
            # the state encoder in IC3 code is not consistent with that described in the paper.
            # Here we follow the impelmentation in the paper.
            xi = tf.expand_dims(tf.reshape(tf.boolean_mask(x, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x[i], axis=0), xi], axis=1)
            si = tf.nn.tanh(tf.matmul(xi, w_ob[i]) + b_ob[i]) + tf.matmul(mi, w_msg[i]) + b_msg[i]
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_ic3_hetero(xs, dones, masks, s, n_s_ls, n_a_ls, scope, init_scale=DEFAULT_SCALE,
                    init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    ns_dim_ls = []
    for i in range(n_agent):
        n_s = n_s_ls[i]
        ns_dim = []
        for j in np.where(masks[i])[0]:
            n_s += n_s_ls[j]
            ns_dim.append(n_s_ls[j])
        n_m = len(ns_dim)
        ns_dim_ls.append(ns_dim)
        with tf.variable_scope(scope + ('_%d' % i)):
            if n_m:
                w_msg.append(tf.get_variable("w_msg", [n_h, n_h],
                                             initializer=init_method(init_scale, init_mode)))
                b_msg.append(tf.get_variable("b_msg", [n_h],
                                             initializer=tf.constant_initializer(0.0)))
            else:
                w_msg.append(None)
                b_msg.append(None)
            w_ob.append(tf.get_variable("w_ob", [n_s, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, done) in enumerate(zip(xs, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        out_h = []
        out_c = []
        out_m = [tf.expand_dims(h[i], axis=0) for i in range(n_agent)]
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            n_m = len(ns_dim_ls[i])
            xi = [tf.slice(x, [i, 0], [1, n_s_ls[i]])]
            if n_m:
                mi = tf.reduce_mean(tf.boolean_mask(out_m, masks[i]), axis=0, keepdims=True)
                raw_xi = tf.boolean_mask(x, masks[i])
                for j in range(n_m):
                    xi.append(tf.slice(raw_xi, [j, 0], [1, ns_dim_ls[i][j]]))
                xi = tf.concat(xi, axis=1)
            else:
                xi = xi[0]
            # the state encoder in IC3 code is not consistent with that described in the paper.
            # Here we follow the impelmentation in the paper.
            si = tf.nn.tanh(tf.matmul(xi, w_ob[i]) + b_ob[i])
            if n_m:
                si = si + tf.matmul(mi, w_msg[i]) + b_msg[i]
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_dial(xs, ps, dones, masks, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
              init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    n_s = xs.shape[-1]
    n_a = ps.shape[-1]
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    for i in range(n_agent):
        n_m = np.sum(masks[i])
        # n_in_hid = (n_m+1)*n_h
        with tf.variable_scope(scope + ('_%d' % i)):
            w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                         initializer=init_method(init_scale, init_mode)))
            b_msg.append(tf.get_variable("b_msg", [n_h],
                                         initializer=tf.constant_initializer(0.0)))
            w_ob.append(tf.get_variable("w_ob", [n_s*(n_m+1), n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        for i in range(n_agent):
            hi = tf.expand_dims(h[i], axis=0)
            mi = fc(hi, 'mfc_%d' % i, n_h)
            out_m.append(mi)
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
            ai = tf.one_hot(tf.expand_dims(tf.argmax(p[i]), axis=0), n_h)
            xi = tf.expand_dims(tf.reshape(tf.boolean_mask(x, masks[i]), [-1]), axis=0)
            xi = tf.concat([tf.expand_dims(x[i], axis=0), xi], axis=1)
            hxi = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
            si = hxi + hmi + ai
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s


def lstm_dial_hetero(xs, ps, dones, masks, s, n_s_ls, n_a_ls, scope, init_scale=DEFAULT_SCALE,
                     init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    n_agent = s.shape[0]
    n_h = s.shape[1] // 2
    xs = tf.transpose(xs, perm=[1,0,2]) # TxNxn_s
    xs = batch_to_seq(xs)
    ps = tf.transpose(ps, perm=[1,0,2]) # TxNxn_a
    ps = batch_to_seq(ps)
    # need dones to reset states
    dones = batch_to_seq(dones) # Tx1
    # create wts
    w_msg = []
    b_msg = []
    w_ob = []
    b_ob = []
    wx_hid = []
    wh_hid = []
    b_hid = []
    ns_dim_ls = []
    for i in range(n_agent):
        n_s = n_s_ls[i]
        ns_dim = []
        for j in np.where(masks[i])[0]:
            n_s += n_s_ls[j]
            ns_dim.append(n_s_ls[j])
        n_m = len(ns_dim)
        ns_dim_ls.append(ns_dim)
        with tf.variable_scope(scope + ('_%d' % i)):
            if n_m:
                w_msg.append(tf.get_variable("w_msg", [n_h*n_m, n_h],
                                             initializer=init_method(init_scale, init_mode)))
                b_msg.append(tf.get_variable("b_msg", [n_h],
                                             initializer=tf.constant_initializer(0.0)))
            else:
                w_msg.append(None)
                b_msg.append(None)
            w_ob.append(tf.get_variable("w_ob", [n_s, n_h],
                                        initializer=init_method(init_scale, init_mode)))
            b_ob.append(tf.get_variable("b_ob", [n_h],
                                        initializer=tf.constant_initializer(0.0)))
            wx_hid.append(tf.get_variable("wx_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            wh_hid.append(tf.get_variable("wh_hid", [n_h, n_h*4],
                                          initializer=init_method(init_scale, init_mode)))
            b_hid.append(tf.get_variable("b_hid", [n_h*4],
                                         initializer=tf.constant_initializer(0.0)))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    # loop over steps
    for t, (x, p, done) in enumerate(zip(xs, ps, dones)):
        # abuse 1 agent as 1 step
        x = tf.squeeze(x, axis=0)
        p = tf.squeeze(p, axis=0)
        out_h = []
        out_c = []
        out_m = []
        # communication phase
        for i in range(n_agent):
            hi = tf.expand_dims(h[i], axis=0)
            mi = fc(hi, 'mfc_%d' % i, n_h)
            out_m.append(mi)
        out_m = tf.concat(out_m, axis=0) # Nxn_h
        # hidden phase
        for i in range(n_agent):
            ci = tf.expand_dims(c[i], axis=0)
            hi = tf.expand_dims(h[i], axis=0)
            # reset states for a new episode
            ci = ci * (1-done)
            hi = hi * (1-done)
            # receive neighbor messages
            n_m = len(ns_dim_ls[i])
            xi = [tf.slice(x, [i, 0], [1, n_s_ls[i]])]
            if n_m:
                mi = tf.expand_dims(tf.reshape(tf.boolean_mask(out_m, masks[i]), [-1]), axis=0)
                ai = tf.one_hot(tf.expand_dims(tf.argmax(p[i]), axis=0), n_h)
                raw_xi = tf.boolean_mask(x, masks[i])
                for j in range(n_m):
                    xi.append(tf.slice(raw_xi, [j, 0], [1, ns_dim_ls[i][j]]))
                xi = tf.concat(xi, axis=1)
            else:
                xi = xi[0]
            si = tf.nn.relu(tf.matmul(xi, w_ob[i]) + b_ob[i])
            if n_m:
                hmi = tf.nn.relu(tf.matmul(mi, w_msg[i]) + b_msg[i])
                si = si + hmi + ai
            zi = tf.matmul(si, wx_hid[i]) + tf.matmul(hi, wh_hid[i]) + b_hid[i]
            ii, fi, oi, ui = tf.split(axis=1, num_or_size_splits=4, value=zi)
            ii = tf.nn.sigmoid(ii)
            fi = tf.nn.sigmoid(fi)
            oi = tf.nn.sigmoid(oi)
            ui = tf.tanh(ui)
            ci = fi*ci + ii*ui
            hi = oi*tf.tanh(ci)
            out_h.append(hi)
            out_c.append(ci)
        c = tf.concat(out_c, axis=0)
        h = tf.concat(out_h, axis=0)
        xs[t] = tf.expand_dims(h, axis=0)
    s = tf.concat(axis=1, values=[c, h])
    xs = seq_to_batch(xs) # TxNxn_h
    xs = tf.transpose(xs, perm=[1,0,2]) # NxTxn_h
    return xs, s

"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        self.gamma = gamma
        self.alpha = alpha
        if alpha > 0:
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.adds = []
        self.dones = [done]

    def add_transition(self, ob, na, a, r, v, done):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        nas = np.array(self.adds, dtype=np.int32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, nas, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        # use post-step dones here
        tdiff = dt
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            if done:
                tdiff = 0
            # additional spatial rewards
            tmax = min(tdiff, self.max_distance)
            for t in range(tmax + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.gamma * self.alpha) ** t * rt
            Adv = R - v
            tdiff += 1
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            # additional spatial rewards
            for t in range(self.max_distance + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.alpha ** t) * rt
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs


class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))
        policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))
        acts = np.transpose(np.array(self.acts, dtype=np.int32))
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, policies, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = r + self.gamma * cur_R * (1.-done)
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            tdiff = dt
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                if done:
                    tdiff = 0
                # additional spatial rewards
                tmax = min(tdiff, max_distance)
                for t in range(tmax + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.gamma * self.alpha) ** t * rt
                cur_Adv = cur_R - v
                tdiff += 1
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                # additional spatial rewards
                for t in range(max_distance + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.alpha ** t) * rt
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val

