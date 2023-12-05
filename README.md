# Deep kinematic inference affords efficient and scalable control of bodily movements

<p align="center">
  <img src="/reference/images/env.png">
</p>

<p align="justify">
Performing goal-directed movements requires mapping goals from extrinsic (workspace-relative) to intrinsic (body-relative) coordinates and then to motor signals. Mainstream approaches based on Optimal Control realize the mappings by minimizing cost functions, which is computationally demanding. Active inference instead uses generative models to produce sensory predictions, which allows computing motor signals through a cheaper inversion of the predictions. However, devising generative models to control complex kinematic plants like the human body is challenging. We introduce a novel Active Inference architecture that affords a simple but effective mapping from extrinsic to intrinsic coordinates via inference and easily scales up to drive complex kinematic plants. Rich goals can be specified in both intrinsic and extrinsic coordinates using both attractive and repulsive forces. The novel deep kinematic inference model reproduces sophisticated bodily movements and paves the way for computationally efficient and biologically plausible control of actuated systems.
</p>

**Reference Paper**: https://www.biorxiv.org/content/10.1101/2023.05.04.539409v1

## How To

<p align="justify">
The simulation can be launched through <b>main.py</b>. either with the option "-m" for manual control, "-s" for the shallow model, "-d" for the deep hierarchical model, "-j" for the standard Jacobian control, or "-a" for choosing the parameters from the console. If no option is specified, the last option will be launched. For the manual control simulation, the keys Z, X, A, S, LEFT, RIGHT, UP and DOWN can be used in order to move the joints of the arm.
</p>

More advanced parameters can be manually set from **config.py**.

<p align="justify">
Plots can be generated through <b>plot.py</b>, either with the option "-d" for the dynamics, "-s" for the score plots, or "-v" for generating a video of the simulation.
</p>

## Required Libraries

matplotlib==3.6.2

numpy==1.24.1

pandas==1.5.3

pyglet==1.5.23

seaborn==0.12.2
