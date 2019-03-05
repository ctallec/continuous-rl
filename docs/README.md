This blog post gives a short summary of the article [Making Deep Q-learning Approaches Robust to Time Discretization](https://arxiv.org/abs/1901.09732).

# A bit of motivation
Have you ever tried training a *Deep Deterministic Policy Gradient* (DDPG TODO{cite})
agent on the *Bipedal Walker* (TODO{cite}) environment? With very little hyperparameterisation,
you can get it to kind of work, and you would probably obtain something of the sort:

![It is not much, but it is at least trying to go forward.](docs/vids/ddpg_high_best.gif)

Now, have you ever tried to do it with a framerate of 1000 FPS instead of the usual 50 FPS?
Kind of crazy, but things should only be easier. We are just providing our agent with a much smaller 
reaction time, and this should only improve its performance. If you were able to react 20 times faster
than you normally, you would expect to become much better at everything you do, not much worse.

![Not quite the results we were expecting...](docs/vids/ddpg_low_best.gif)

Strange, the agent is not learning anything anymore... And if you perform the same experiment
on different environments, you will get the same kind of results. There seems to be something
wrong with Q-learning when the framerate time becomes arbitrarily high. Time to put on
the detective cap and investigate!

# A crash course on approximate Q-learning
Before going into the depth of why Q-learning is failing when the framerate becomes high,
let's give ...

As a reinforcement learning algorithm, Q-learning displays three key properties:
	1. It is *temporal difference based*. More precisely it revolves around
the use of the *optimal Bellman Equation*.
	2. It is *model free*. It does not need to know, or to model the
dynamic of the environment to learn.
	3. It is *off-policy*, i.e. it learns a different policy than the one
it actually uses to produce its training trajectories.

# What is continuous time Reinforcement Learning, and why does it matters

# What's wrong with near continuous Q-learning?

# Can we fix it?

# Results

# Outlines
