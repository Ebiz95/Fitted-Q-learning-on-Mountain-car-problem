# Reinforcement learning with a bilinear Q function

This paper, [Reinforcement learning with a bilinear Q function](https://github.com/Ebiz95/Fitted-Q-learning-on-Mountain-car-problem/blob/main/Reinforcement%20learning%20with%20a%20bilinear%20Q%20function.pdf), represents the Q fucntion as `Q(s,a) = s^T * W * a`, where **s** represents the state-space, **a** is the action-space and **W** is a learned matrix. **W** is learned through linear regression using a batch of examples sampled from the environment. One of the advantages of this method is that it can be done offline, without any agent-environment interaction. Also, the number of learned parameters in **W** is considerably low.

## Mountain Car Problem

The python implementation on the [continuous Mountain Car](https://gym.openai.com/envs/MountainCarContinuous-v0/) problem implemented in the [Gym](https://gym.openai.com/) library can be found [here](https://github.com/Ebiz95/Fitted-Q-learning-on-Mountain-car-problem).

The program can be started from the Main.py file. At the top of the file there are certain parameters and flags that you can set. It is up to you whether you want to test many parameters at the same time or just one set of parameters. It is also possible to render the video to see how the algorithm performs by setting the *render_video* flag to True. Note that the samples are generated randomly at the start of training, which means that it is possible to end up with a bad batch of samples that does not actually lead to a solution.
