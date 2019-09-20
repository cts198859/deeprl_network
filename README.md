# Networked Multi-agent RL (NMARL)
This repo implements the state-of-the-art MARL algorithms for networked system control, with observability and communication of each agent limited to its neighborhood. For fair comparison, all algorithms are applied to A2C agents, and they are classified into two groups. IA2C contains non-communicative policies which utilize neighborhood information only, whereas MA2C contains communicative policies with certain communication protocols.

Available IA2C algorithms:
* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)


Available MA2C algorithms:
* DIAL: [Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf)
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: Inspired from [Gilmer, Justin, et al. "Neural message passing for quantum chemistry." arXiv preprint arXiv:1704.01212, 2017.](https://arxiv.org/pdf/1704.01212.pdf)

Available NMARL scenaios:
* ATSC Grid: Adaptive traffic signal control in a synthetic traffic grid.
* ATSC Monaco: Adaptive traffic signal control in a real-world traffic network from Monaco city.
* CACC Catch-up: Cooperative adaptive cruise control for catching up the leadinig vehicle.
* CACC Slow-down: Cooperative adaptive cruise control for following the leading vehicle to slow down.

## Requirements
* Python3
* [Tensorflow](http://www.tensorflow.org/install)
* [SUMO](http://sumo.dlr.de/wiki/Installing)


