"""
Main function for training and evaluating MARL algorithms in NMARL envs
@author: Tianshu Chu
"""

import argparse
import configparser
import logging
import threading
from torch.utils.tensorboard.writer import SummaryWriter
from envs.cacc_env import CACCEnv
from envs.large_grid_env import LargeGridEnv
from envs.real_net_env import RealNetEnv
from agents.models import IA2C
# from agents.models import IA2C, IA2C_FP, IA2C_CU, MA2C_NC, MA2C_IC3, MA2C_DIAL
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag)


def parse_args():
    default_base_dir = '/Users/ctianshu/Documents/deeprl_net/ia2c_grid_0.9'
    default_config_dir = './config/config_ia2c_grid.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(2000, 2500, 10)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0):
    scenario = config.get('scenario')
    if scenario.startswith('atsc'):
        if scenario.endswith('large_grid'):
            return LargeGridEnv(config, port=port)
        else:
            return RealNetEnv(config, port=port)
    else:
        return CACCEnv(config)


def init_agent(env, config, total_step, seed):
    if env.agent == 'ia2c':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc':
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_ic3':
        # this is actually CommNet
        return MA2C_IC3(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                        total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu':
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial':
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'])
    logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    model = init_agent(env, config['MODEL_CONFIG'], total_step, seed)
    model.load(dirs['model'], train_mode=True)

    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    trainer = Trainer(env, model, global_counter, summary_writer, output_path=dirs['data'])
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()


def evaluate_fn(agent_dir, output_dir, seeds, port, demo):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file 
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
