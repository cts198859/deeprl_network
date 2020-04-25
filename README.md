# Networked Multi-agent RL (NMARL)
This repo implements the state-of-the-art MARL algorithms for networked system control, with observability and communication of each agent limited to its neighborhood. For fair comparison, all algorithms are applied to A2C agents, classified into two groups: IA2C contains non-communicative policies which utilize neighborhood information only, whereas MA2C contains communicative policies with certain communication protocols.

Available IA2C algorithms:
* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)


Available MA2C algorithms:
* DIAL: [Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf)
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: Inspired from [Gilmer, Justin, et al. "Neural message passing for quantum chemistry." arXiv preprint arXiv:1704.01212, 2017.](https://arxiv.org/pdf/1704.01212.pdf)

Available NMARL scenarios:
* ATSC Grid: Adaptive traffic signal control in a synthetic traffic grid.
* ATSC Monaco: Adaptive traffic signal control in a real-world traffic network from Monaco city.
* CACC Catch-up: Cooperative adaptive cruise control for catching up the leadinig vehicle.
* CACC Slow-down: Cooperative adaptive cruise control for following the leading vehicle to slow down.

## Requirements
* Python3 == 3.5
* [PyTorch](https://pytorch.org/get-started/locally/) == 1.4.0
* [Tensorflow](http://www.tensorflow.org/install) == 2.1.0 (for tensorboard) 
* [SUMO](http://sumo.dlr.de/wiki/Installing) >= 1.1.0

## Usages
First define all hyperparameters (including algorithm and DNN structure) in a config file under `[config_dir]` ([examples](./config)), and create the base directory of each experiement `[base_dir]`. For ATSC Grid, please call [`build_file.py`](./envs/large_grid_data) to generate SUMO network files before training.

1. To train a new agent, run
~~~
python3 main.py --base-dir [base_dir] train --config-dir [config_dir]
~~~
Training config/data and the trained model will be output to `[base_dir]/data` and `[base_dir]/model`, respectively.

2. To access tensorboard during training, run
~~~
tensorboard --logdir=[base_dir]/log
~~~

3. To evaluate a trained agent, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seeds]
~~~
Evaluation data will be output to `[base_dir]/eva_data`. Make sure evaluation seeds are different from those used in training.    

4. To visualize the agent behavior in ATSC scenarios, run
~~~
python3 main.py --base-dir [base_dir] evaluate --evaluation-seeds [seed] --demo
~~~
It is recommended to use only one evaluation seed for the demo run. This will launch the SUMO GUI, and [`view.xml`](./envs/large_grid_data) can be applied to visualize queue length and intersectin delay in edge color and thickness. 

## Reproducibility
The paper results are based on an out-of-date SUMO version 0.32.0. We are re-running the experiments with SUMO 1.2.0 and will update the results soon. The pytorch impelmention is avaliable at branch [pytorch](https://github.com/cts198859/deeprl_network/tree/pytorch).

## Citation
For more implementation details and underlying reasonings, please check our paper [Multi-agent Reinforcement Learning for Networked System Control](https://openreview.net/forum?id=Syx7A3NFvH).
~~~
@inproceedings{
chu2020multiagent,
title={Multi-agent Reinforcement Learning for Networked System Control},
author={Tianshu Chu and Sandeep Chinchali and Sachin Katti},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Syx7A3NFvH}
}
~~~


