# Playing MsPac-man with Deep Reinforcement Learning

The first version of the Deep Q-Network by DeepMind (Huang, 2013) was capable of human-level performance on a number of classic Atari 2600 games. The algorithm used a CNN architecture which based its strategy from vision, much like a human player. Training from scratch with no prior knowledge of the environment, it discovered strategies that enabled it to exceed human benchmarks. Since then, many refinements and optimisations have been attempted, and we aim to benchmark some of the recent advancements with the MsPac-man environment from the python package Gym (Brockman et al., 2016).