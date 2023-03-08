# Latent-Conditioned Policy Gradient for Multi-Objective Deep Reinforcement Learning
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

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).
