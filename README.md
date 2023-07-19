# Latent-Conditioned Policy Gradient for Multi-Objective Deep Reinforcement Learning
[https://arxiv.org/abs/2303.08909](https://arxiv.org/abs/2303.08909)

Accepted by [the 32nd International Conference on Artificial Neural Networks (ICANN2023)](https://e-nns.org/icann2023/), 
Crete, Greece from 26th to 29th of September 2023.

## Authors
Takuya Kanazawa (Hitachi, Ltd.) and Chetan Gupta (Hitachi America, Ltd.)

## Abstract
Sequential decision making in the real world often requires finding a good balance of conflicting
objectives. In general, there exist a plethora of Pareto-optimal policies that embody different
patterns of compromises between objectives, and it is technically challenging to obtain them
exhaustively using deep neural networks. In this work, we propose a novel multi-objective reinforcement
learning (MORL) algorithm that trains a single neural network via policy gradient to
approximately obtain the entire Pareto set in a single run of training, without relying on linear
scalarization of objectives. The proposed method works in both continuous and discrete action spaces
with no design change of the policy network. Numerical experiments in benchmark environments
demonstrate the practicality and efficacy of our approach in comparison to standard MORL baselines.

## License
Copyright 2023 LCMOPG Authors and Hitachi, Ltd.

Licensed under the Apache License, Version 2.0 (the "License"): [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).
