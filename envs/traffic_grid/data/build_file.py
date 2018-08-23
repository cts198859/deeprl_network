# -*- coding: utf-8 -*-
"""
build *.xml files for a large 5 x 5 network
w/ the traffic dynamics modified from the following paper:

Chu, Tianshu, Shuhui Qu, and Jie Wang. "Large-scale traffic grid signal control with
regional reinforcement learning." American Control Conference (ACC), 2016. IEEE, 2016.

@author: Tianshu Chu
"""
import numpy as np
import os

SPEED_LIMIT = 20
L0 = 200
L0_end = 75
N = 5


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def output_nodes(node):
    str_nodes = '<nodes>\n'
    # traffic light nodes
    ind = 1
    for dy in np.arange(0, L0 * 5, L0):
        for dx in np.arange(0, L0 * 5, L0):
            str_nodes += node % ('nt' + str(ind), dx, dy, 'traffic_light')
            ind += 1
    # other nodes
    ind = 1
    for dx in np.arange(0, L0 * 5, L0):
        str_nodes += node % ('np' + str(ind), dx, -L0_end, 'priority')
        ind += 1
    for dy in np.arange(0, L0 * 5, L0):
        str_nodes += node % ('np' + str(ind), L0 * 4 + L0_end, dy, 'priority')
        ind += 1
    for dx in np.arange(L0 * 4, -1, -L0):
        str_nodes += node % ('np' + str(ind), dx, L0 * 4 + L0_end, 'priority')
        ind += 1
    for dy in np.arange(L0 * 4, -1, -L0):
        str_nodes += node % ('np' + str(ind), -L0_end, dy, 'priority')
        ind += 1
    str_nodes += '</nodes>\n'
    return str_nodes


def output_road_types():
    str_types = '<types>\n'
    str_types += '  <type id="a" priority="2" numLanes="1" speed="%.2f"/>\n' % SPEED_LIMIT
    str_types += '  <type id="b" priority="1" numLanes="1" speed="%.2f"/>\n' % SPEED_LIMIT
    str_types += '</types>\n'
    return str_types


def get_edge_str(edge, from_node, to_node, edge_type):
    edge_id = 'e:%s,%s' % (from_node, to_node)
    return edge % (edge_id, from_node, to_node, edge_type)


def output_edges(edge):
    str_edges = '<edges>\n'
    # external roads
    in_edges = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1]
    out_edges = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
    for in_i, out_i in zip(in_edges, out_edges):
        in_node = 'nt' + str(in_i)
        out_node = 'np' + str(out_i)
        str_edges += get_edge_str(edge, in_node, out_node, 'a')
        str_edges += get_edge_str(edge, out_node, in_node, 'a')

    in_edges = [1, 2, 3, 4, 5, 25, 24, 23, 22, 21]
    out_edges = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    for in_i, out_i in zip(in_edges, out_edges):
        in_node = 'nt' + str(in_i)
        out_node = 'np' + str(out_i)
        str_edges += get_edge_str(edge, in_node, out_node, 'b')
        str_edges += get_edge_str(edge, out_node, in_node, 'b')
    # internal roads
    for i in range(1, 25, 5):
        for j in range(4):
            from_node = 'nt' + str(i + j)
            to_node = 'nt' + str(i + j + 1)
            str_edges += get_edge_str(edge, from_node, to_node, 'a')
            str_edges += get_edge_str(edge, to_node, from_node, 'a')
    for i in range(1, 6):
        for j in range(0, 20, 5):
            from_node = 'nt' + str(i + j)
            to_node = 'nt' + str(i + j + 5)
            str_edges += get_edge_str(edge, from_node, to_node, 'b')
            str_edges += get_edge_str(edge, to_node, from_node, 'b')
    str_edges += '</edges>\n'
    return str_edges


def get_con_str(con, from_node, cur_node, to_node):
    from_edge = 'e:%s,%s' % (from_node, cur_node)
    to_edge = 'e:%s,%s' % (cur_node, to_node)
    return con % (from_edge, to_edge)


def get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node):
    str_cons = ''
    # go-through
    str_cons += get_con_str(con, s_node, cur_node, n_node)
    str_cons += get_con_str(con, n_node, cur_node, s_node)
    str_cons += get_con_str(con, w_node, cur_node, e_node)
    str_cons += get_con_str(con, e_node, cur_node, w_node)
    # left-turn
    str_cons += get_con_str(con, s_node, cur_node, w_node)
    str_cons += get_con_str(con, n_node, cur_node, e_node)
    str_cons += get_con_str(con, w_node, cur_node, n_node)
    str_cons += get_con_str(con, e_node, cur_node, s_node)
    # right-turn
    str_cons += get_con_str(con, s_node, cur_node, e_node)
    str_cons += get_con_str(con, n_node, cur_node, w_node)
    str_cons += get_con_str(con, w_node, cur_node, s_node)
    str_cons += get_con_str(con, e_node, cur_node, n_node)
    return str_cons


def output_connections(con):
    str_cons = '<connections>\n'
    # edge nodes
    in_edges = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1]
    out_edges = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
    for i, j in zip(in_edges, out_edges):
        if i == 5:
            s_node = 'np5'
        elif i == 1:
            s_node = 'np1'
        else:
            s_node = 'nt' + str(i - 5)
        if i == 25:
            n_node = 'np11'
        elif i == 21:
            n_node = 'np15'
        else:
            n_node = 'nt' + str(i + 5)
        if i % 5 == 1:
            w_node = 'np' + str(j)
        else:
            w_node = 'nt' + str(i - 1)
        if i % 5 == 0:
            e_node = 'np' + str(j)
        else:
            e_node = 'nt' + str(i + 1)
        cur_node = 'nt' + str(i)
        str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    in_edges = [2, 3, 4, 24, 23, 22]
    out_edges = [2, 3, 4, 12, 13, 14]
    for i, j in zip(in_edges, out_edges):
        w_node = 'nt' + str(i - 1)
        e_node = 'nt' + str(i + 1)
        if i <= 5:
            s_node = 'np' + str(j)
        else:
            s_node = 'nt' + str(i - 5)
        if i >= 20:
            n_node = 'np' + str(j)
        else:
            n_node = 'nt' + str(i + 5)
        cur_node = 'nt' + str(i)
        str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    # internal nodes
    for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
        n_node = 'nt' + str(i + 5)
        s_node = 'nt' + str(i - 5)
        w_node = 'nt' + str(i - 1)
        e_node = 'nt' + str(i + 1)
        cur_node = 'nt' + str(i)
        str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    str_cons += '</connections>\n'
    return str_cons


def output_netconfig():
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <edge-files value="exp.edg.xml"/>\n'
    str_config += '    <node-files value="exp.nod.xml"/>\n'
    str_config += '    <type-files value="exp.typ.xml"/>\n'
    str_config += '    <tllogic-files value="exp.tll.xml"/>\n'
    str_config += '    <connection-files value="exp.con.xml"/>\n'
    str_config += '  </input>\n  <output>\n'
    str_config += '    <output-file value="exp.net.xml"/>\n'
    str_config += '  </output>\n</configuration>\n'
    return str_config


def get_external_od(out_edges, dest=True):
    edge_maps = [0, 1, 2, 3, 4, 5, 5, 10, 15, 20, 25,
                 25, 24, 23, 22, 21, 21, 16, 11, 6, 1]
    cur_dest = []
    for out_edge in out_edges:
        in_edge = edge_maps[out_edge]
        in_node = 'nt' + str(in_edge)
        out_node = 'np' + str(out_edge)
        if dest:
            edge = 'e:%s,%s' % (in_node, out_node)
        else:
            edge = 'e:%s,%s' % (out_node, in_node)
        cur_dest.append(edge)
    return cur_dest


def sample_od_pair(orig_edges, dest_edges):
    from_edges = []
    to_edges = []
    for i in range(len(orig_edges)):
        from_edges.append(np.random.choice(orig_edges[i]))
        to_edges.append(np.random.choice(dest_edges))
    return from_edges, to_edges


def output_flows(num_ext_car_hourly, num_int_car_hourly, seed=None):
    if seed is not None:
        np.random.seed(seed)
    prob = num_int_car_hourly / 3600
    ext_flow = '  <flow id="fe:%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    int_flow = '  <flow id="fi:%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" probability="%.2f" type="type1"/>\n'
    str_flows = '<routes>\n'
    str_flows += '  <vType id="type1" length="5" accel="5" decel="10"/>\n'
    # create internal origins and destinations
    origs = []
    for i in [17, 19, 7, 9]:
        cur_orig = []
        n_node = 'nt' + str(i + 5)
        s_node = 'nt' + str(i - 5)
        w_node = 'nt' + str(i - 1)
        e_node = 'nt' + str(i + 1)
        cur_node = 'nt' + str(i)
        for next_node in [n_node, s_node, w_node, e_node]:
            edge1 = 'e:%s,%s' % (next_node, cur_node)
            edge2 = 'e:%s,%s' % (cur_node, next_node)
            cur_orig.append([edge1, edge2])
        origs.append(cur_orig)
    origs = np.array(origs)

    dests = []
    dests.append(get_external_od([3, 4, 5]))
    dests.append(get_external_od([6, 7, 8]))
    dests.append(get_external_od([8, 9, 10]))
    dests.append(get_external_od([11, 12, 13]))
    dests.append(get_external_od([13, 14, 15]))
    dests.append(get_external_od([16, 17, 18]))
    dests.append(get_external_od([18, 19, 20]))
    dests = np.array(dests)

    times = range(0, 7201, 1200)
    orig_inds = [[0, 3], [0, 1], [1, 2], [2, 3], [0, 3], [1, 2]]
    dest_inds = [[2, 6], [2, 6], [6, 3], [4, 0], [0, 4], [5, 1]]

    # create external origins and destinations
    srcs = []
    srcs.append(get_external_od([1, 2, 3, 13, 14, 15], dest=False))
    srcs.append(get_external_od([2, 3, 4, 16, 17, 18], dest=False))
    srcs.append(get_external_od([3, 4, 5, 17, 18, 19], dest=False))
    srcs.append(get_external_od([6, 7, 8, 18, 19, 20], dest=False))
    srcs.append(get_external_od([1, 2, 3, 7, 8, 9], dest=False))
    srcs.append(get_external_od([2, 3, 4, 8, 9, 10], dest=False))

    sinks = []
    sinks.append(get_external_od([13, 12, 11, 5, 4, 3]))
    sinks.append(get_external_od([14, 13, 12, 8, 7, 6]))
    sinks.append(get_external_od([15, 14, 13, 9, 8, 7]))
    sinks.append(get_external_od([18, 17, 16, 10, 9, 8]))
    sinks.append(get_external_od([13, 12, 11, 19, 18, 17]))
    sinks.append(get_external_od([14, 13, 12, 20, 19, 18]))

    for t in range(len(times) - 1):
        name = str(t)
        t_begin, t_end = times[t], times[t + 1]
        # internal flow
        k = 0
        for i, j in zip(orig_inds[t], dest_inds[t]):
            from_edges, to_edges = sample_od_pair(origs[i], dests[j])
            for e1, e2 in zip(from_edges, to_edges):
                cur_name = name + '_' + str(k)
                str_flows += int_flow % (cur_name, e1, e2, t_begin, t_end, prob)
                k += 1
        # external flow
        k = 0
        for e1, e2 in zip(srcs[t], sinks[t]):
            cur_name = name + '_' + str(k)
            str_flows += ext_flow % (cur_name, e1, e2, t_begin, t_end, num_ext_car_hourly)
            k += 1
    str_flows += '</routes>\n'
    return str_flows


def gen_rou_file(path, num_ext_car_hourly, num_int_car_hourly, seed=None, thread=None):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)
    write_file(path + flow_file, output_flows(num_ext_car_hourly, num_int_car_hourly, seed=seed))
    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file


def output_config(thread=None):
    if thread is None:
        out_file = 'exp.rou.xml'
    else:
        out_file = 'exp_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="exp.net.xml"/>\n'
    str_config += '    <route-files value="%s"/>\n' % out_file
    str_config += '    <additional-files value="exp.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="7200"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config


def get_ild_str(from_node, to_node, ild_in=None, ild_out=None):
    edge = '%s,%s' % (from_node, to_node)
    ild_str = ''
    if ild_in:
        ild_str += ild_in % (edge, 'e:' + edge)
    if ild_out:
        ild_str += ild_out % (edge, 'e:' + edge)
    return ild_str


def output_ild(ild_in, ild_out):
    str_adds = '<additional>\n'
    in_edges = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1,
                1, 2, 3, 4, 5, 25, 24, 23, 22, 21]
    out_edges = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20,
                 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    # external edges
    for i, j in zip(in_edges, out_edges):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        str_adds += get_ild_str(node1, node2, ild_in=ild_in)
        str_adds += get_ild_str(node2, node1, ild_out=ild_out)
    # streets
    for i in range(1, 25, 5):
        for j in range(4):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 1)
            str_adds += get_ild_str(node1, node2, ild_in=ild_in, ild_out=ild_out)
            str_adds += get_ild_str(node2, node1, ild_in=ild_in, ild_out=ild_out)
    # avenues
    for i in range(1, 6):
        for j in range(0, 20, 5):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 5)
            str_adds += get_ild_str(node1, node2, ild_in=ild_in, ild_out=ild_out)
            str_adds += get_ild_str(node2, node1, ild_in=ild_in, ild_out=ild_out)
    str_adds += '</additional>\n'
    return str_adds


def output_tls(tls, phase):
    str_adds = '<additional>\n'
    # all crosses have 2 phases
    two_phases = ['GGgrrrGGgrrr', 'yyyrrryyyrrr',
                  'rrrGGgrrrGGg', 'rrryyyrrryyy']
    phase_duration = [30, 3]
    for i in range(1, 26):
        node = 'nt' + str(i)
        str_adds += tls % node
        for k, p in enumerate(two_phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'
    str_adds += '</additional>\n'
    return str_adds


def main():
    # nod.xml file
    node = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
    write_file('./exp.nod.xml', output_nodes(node))

    # typ.xml file
    write_file('./exp.typ.xml', output_road_types())

    # edg.xml file
    edge = '  <edge id="%s" from="%s" to="%s" type="%s"/>\n'
    write_file('./exp.edg.xml', output_edges(edge))

    # con.xml file
    con = '  <connection from="%s" to="%s" fromLane="0" toLane="0"/>\n'
    write_file('./exp.con.xml', output_connections(con))

    # tls.xml file
    tls = '  <tlLogic id="%s" programID="0" offset="0" type="static">\n'
    phase = '    <phase duration="%d" state="%s"/>\n'
    write_file('./exp.tll.xml', output_tls(tls, phase))

    # net config file
    write_file('./exp.netccfg', output_netconfig())

    # generate net.xml file
    os.system('netconvert -c exp.netccfg')

    # raw.rou.xml file
    write_file('./exp.rou.xml', output_flows(1000, 2000))

    # generate rou.xml file
    # os.system('jtrrouter -n exp.net.xml -r exp.raw.rou.xml -o exp.rou.xml')

    # add.xml file
    ild_out = '  <inductionLoop file="ild_in.out" freq="15" id="ild_out:%s" lane="%s_0" pos="-50"/>\n'
    ild_in = '  <inductionLoop file="ild_out.out" freq="15" id="ild_in:%s" lane="%s_0" pos="10"/>\n'
    write_file('./exp.add.xml', output_ild(ild_in, ild_out))

    # config file
    write_file('./exp.sumocfg', output_config())

if __name__ == '__main__':
    main()
