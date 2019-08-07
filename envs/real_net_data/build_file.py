import configparser
import logging
import numpy as np
import os

# from envs.real_net_env import RealNetEnv

ILD_POS = 50

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def output_flows(flow_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
    FLOW_NUM = 6
    flows = []
    flows1 = []
    # flows1.append(('-10114#1', '-10079', '10115#2 -10109 10089#3 -10116'))
    # flows1.append(('-10114#1', '-10079', '-10114#0 10108#0 10108#5 -10090#1 gneE18'))
    # flows1.append(('-10114#1', '-10079', '-10114#0 10108#0 10108#5 gneE5 gneE18'))
    # flows1.append(('-10114#1', '10076', '-10114#0 10108#0 -10067#1 gneE9 gneE18'))
    # flows1.append(('-10114#1', '10076', '-10114#0 10107 10080#0 gneE12 10102'))
    # flows1.append(('-10114#1', '10180#1', '-10114#0 10108#0 -10104 10115#5 -10090#1'))
    flows1.append(('-10114#1', '-10079', '10115#2 -10109'))
    flows1.append(('-10114#1', '-10079', '-10114#0 10108#0 gneE5'))
    flows1.append(('-10114#1', '-10079', '-10114#0 10108#0 10102'))
    flows1.append(('-10114#1', '10076', '-10114#0 10107 10102'))
    flows.append(flows1)
    flows1 = []
    # flows1.append(('10096#1', '10063', '10089#3 10091 gneE12 -10065#2'))
    # flows1.append(('10096#1', '10063', '10089#3 gneE4 -10090#1 gneE10'))
    # flows1.append(('-10095', '-10071#3', '10109 10106#3 10115#5 -10080#0'))
    # flows1.append(('-10185#1', '-10071#3', 'gneE20 gneE13 -10046#0 -10090#1'))
    # flows1.append(('-10185#1', '-10061#5', 'gneE19 -10046#5 10089#4 gneE12'))
    # flows1.append(('10197#1', '-10061#5', '10089#3 -10049 10043 10053#0'))
    flows1.append(('10096#1', '10063', '10089#3'))
    flows1.append(('-10185#1', '-10071#3', 'gneE20'))
    flows1.append(('10096#1', '10063', '10109'))
    flows1.append(('-10185#1', '-10061#5', 'gneE19'))
    flows.append(flows1)
    flows1 = []
    # flows1.append(('10052#1', '10104', '10181#1 10116 -10089#3 10109'))
    # flows1.append(('10052#1', '10104', '10181#1 -10089#4 gneE4 gneE7'))
    # flows1.append(('-10051#2', '10043', '10179 10181#1 10116 -10089#3 10109'))
    # flows1.append(('-10051#2', '10043', '10179 10181#1 -10089#4 gneE4 gneE7'))
    # flows1.append(('-10051#2', '-10110', '-10051#0 10181#1 -10089#4 gneE4 -10115#5'))
    # flows1.append(('-10051#2', '-10110', '-10051#0 10181#1 -10089#3 -10049'))
    flows1.append(('10052#1', '10104', '10181#1 -10089#3'))
    flows1.append(('-10064#9', '10104', '-10068 10102'))
    flows1.append(('-10051#2', '10043', '10181#1 gneE4'))
    flows1.append(('-10064#9', '-10110', '-10064#4 -10064#3'))

    flows.append(flows1)
    flows1 = []
    # flows1.append(('-10064#9', '-10085', '-10068 -10064#3 gneE5 10046#0'))
    # flows1.append(('-10064#9', '10085', '-10064#4 -10064#3 gneE5 10046#0'))
    # flows1.append(('-10064#9', '-10086', '-10064#4 10102 10031#1 10046#0'))
    # flows1.append(('10061#4', '-10085', '10065#2 10102 10031#1 10046#0'))
    # flows1.append(('10069#0', '10085', '10065#2 -10064#3 gneE5 10046#0'))
    # flows1.append(('-10058#0', '-10086', '10071#5 10108#5 gneE5 10046#0'))
    flows1.append(('10061#4', '-10085', '10065#2 10102'))
    flows1.append(('10071#3', '10085', '10065#2 -10064#3'))
    flows1.append(('-10070#1', '-10086', 'gneE9'))
    flows1.append(('-10063', '10085', 'gneE8'))
    flows.append(flows1)

    # vols_a = [2, 3, 4, 6, 4, 2, 1, 0, 0, 0, 0]
    # vols_b = [0, 0, 0, 1, 2, 3, 5, 4, 3, 2, 1]
    vols_a = [1, 2, 4, 4, 4, 4, 2, 1, 0, 0, 0]
    vols_b = [0, 0, 0, 1, 2, 4, 4, 4, 4, 2, 1]
    times = np.arange(0, 3301, 300)

    flow_str = '  <flow id="f%s" departPos="random_free" from="%s" to="%s" via="%s" begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
    output = '<routes>\n'
    output += '  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1"/>\n'

    for i in range(len(times) - 1):
        name = str(i)
        t_begin, t_end = times[i], times[i + 1]
        k = 0
        for j in [0, 1]:
            vol = vols_a[i]
            if vol > 0:
                # inds = np.random.choice(FLOW_NUM, vol, replace=False)
                inds = np.arange(vol)
                for ind in inds:
                    cur_name = name + '_' + str(k)
                    src, sink, via = flows[j][ind]
                    output += flow_str % (cur_name, src, sink, via, t_begin, t_end, flow_rate)
                    k += 1
        for j in [2, 3]:
            vol = vols_b[i]
            if vol > 0:
                # inds = np.random.choice(FLOW_NUM, vol, replace=False)
                inds = np.arange(vol)
                for ind in inds:
                    cur_name = name + '_' + str(k)
                    src, sink, via = flows[j][ind]
                    output += flow_str % (cur_name, src, sink, via, t_begin, t_end, flow_rate)
                    k += 1
    output += '</routes>\n'
    return output


def output_config(thread=None):
    if thread is None:
        out_file = 'most.rou.xml'
    else:
        out_file = 'most_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="in/most.net.xml"/>\n'
    str_config += '    <route-files value="in/%s"/>\n' % out_file
    # str_config += '    <additional-files value="in/most.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config


def gen_rou_file(path, flow_rate, seed=None, thread=None):
    if thread is None:
        flow_file = 'most.rou.xml'
    else:
        flow_file = 'most_%d.rou.xml' % int(thread)
    write_file(path + 'in/' + flow_file, output_flows(flow_rate, seed=seed))
    sumocfg_file = path + ('most_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file


def output_ild(env, ild):
    str_adds = '<additional>\n'
    for node_name in env.node_names:
        node = env.nodes[node_name]
        for ild_name in node.ilds_in:
            # ild_name = ild:lane_name
            lane_name = ild_name
            l_len = env.sim.lane.getLength(lane_name)
            i_pos = min(ILD_POS, l_len - 1)
            if lane_name in ['gneE4_0', 'gneE5_0']:
                str_adds += ild % (ild_name, lane_name, -63, -13)
            elif lane_name == 'gneE18_0':
                str_adds += ild % (ild_name, lane_name, -116, -66)
            elif lane_name == 'gneE19_0':
                str_adds += ild % (ild_name, lane_name, 1, 50)
            else:
                str_adds += ild % (ild_name, lane_name, -i_pos, -1)
    str_adds += '</additional>\n'
    return str_adds


# if __name__ == '__main__':
#     logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
#                         level=logging.INFO)
#     config = configparser.ConfigParser()
#     config.read('./config/config_ia2c_real.ini')
#     base_dir = './output_result/'
    # if not os.path.exists(base_dir):
    #     os.mkdir(base_dir)
    # env = RealNetEnv(config['ENV_CONFIG'])
    # add.xml file
    # ild = '  <laneAreaDetector file="ild.out" freq="1" id="%s" lane="%s" pos="%d" endPos="%d"/>\n'
    # write_file('./envs/real_net_data/in/most.add.xml', output_ild(env, ild))
    # env.terminate()
