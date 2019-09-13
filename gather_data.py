import os
import subprocess
import glob

exp_name = 'slowdown'
base_dir = '/home/tchu/rl_test/deeprl_dist'
n_base = len(base_dir.split('/'))
out_dir = os.path.join(base_dir, exp_name + '_train')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
files = glob.glob(os.path.join(base_dir, exp_name + '_*' + 
                               '/data/train_reward.csv'))
for f in files:
    fname = '_'.join(f.split('/')[n_base].split('_')[1:])
    fname = fname + '.csv'
    cmd = 'sudo cp {} {}/{}'.format(f, out_dir, fname)
    subprocess.check_call(cmd, shell=True)