"""
Main function for training and evaluating MARL algorithms in traffic envs
@author: Tianshu Chu
"""

import argparse
import configparser
import logging
import tensorflow as tf
import threading
from envs.large_grid_env import LargeGridEnv, LargeGridController
from agents.models import IA2C, IA2C_FP, MA2C_NC
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag,
                   plot_evaluation, plot_train)


def parse_args():
    default_base_dir = '/Users/tchu/Documents/rl_test/deeprl_dist/ma2c_nc_test'
    default_config_dir = './config/config_ma2c_nc.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--test-mode', type=str, required=False,
                    default='after_train_test',
                    help="test mode during training",
                    choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--agents', type=str, required=False,
                    default='naive', help="agent folder names for evaluation, split by ,")
    sp.add_argument('--evaluate-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(10000, 100001, 10000)]),
                    help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0, naive_policy=False):
    if not naive_policy:
        return LargeGridEnv(config, port=port)
    else:
        env = LargeGridEnv(config, port=port)
        policy = LargeGridController(env.node_names)
        return env, policy


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    in_test, post_test = init_test_flag(args.test_mode)

    # init env
    env = init_env(config['ENV_CONFIG'])
    logging.info('Training: a dim %d, agent dim: %d' % (env.n_a, env.n_agent))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')

    if env.agent == 'ia2c':
        model = IA2C(env.n_s_ls, env.n_a, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                     total_step, config['MODEL_CONFIG'], seed=seed)
    elif env.agent == 'ia2c_fp':
        model = IA2C_FP(env.n_s_ls, env.n_a, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                        total_step, config['MODEL_CONFIG'], seed=seed)
    elif env.agent == 'ma2c_nc':
        model = MA2C_NC(env.n_s, env.n_a, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                        total_step, config['MODEL_CONFIG'], seed=seed)
    else:
        model = None

    # disable multi-threading for safe SUMO implementation
    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = Trainer(env, model, global_counter, summary_writer, in_test, output_path=dirs['data'])
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)

    # post-training test
    if post_test:
        test_dirs = init_dir(base_dir, pathes=['eva_data'])
        evaluator = Evaluator(env, model, test_dirs['eva_data'])
        evaluator.run()


def evaluate_fn(agent_dir, output_dir, seeds, port):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file for env
    config_dir = find_file(agent_dir)
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env, greedy_policy = init_env(config['ENV_CONFIG'], port=port, naive_policy=True)
    env.init_test_seeds(seeds)

    # load model for agent
    if agent != 'greedy':
        # init centralized or multi agent
        if env.agent == 'ia2c':
            model = IA2C(env.n_s_ls, env.n_a, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         0, config['MODEL_CONFIG'])
        elif env.agent == 'ia2c_fp':
            model = IA2C_FP(env.n_s_ls, env.n_a, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                            0, config['MODEL_CONFIG'])
        elif env.agent == 'ma2c_nc':
            model = MA2C_NC(env.n_s, env.n_a, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                            0, config['MODEL_CONFIG'])
        else:
            return
        if not model.load(agent_dir + '/'):
            return
    else:
        model = greedy_policy
    env.agent = agent
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
    init_log(dirs['eva_log'])
    agents = args.agents.split(',')
    # enforce the same evaluation seeds across agents
    seeds = args.evaluate_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    threads = []
    for i, agent in enumerate(agents):
        agent_dir = base_dir + '/' + agent
        thread = threading.Thread(target=evaluate_fn,
                                  args=(agent_dir, dirs['eva_data'], seeds, i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
