# Deep kinematic inference

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the paper [Deep kinematic inference affords efficient and scalable control of bodily movements](https://www.biorxiv.org/content/10.1101/2023.05.04.539409v2). It describes an active inference model that affords a simple but effective mapping from extrinsic to intrinsic coordinates via inference, and easily scales up to drive complex kinematic chains. The proposed model can realize a variety of tasks such as tracking a target while avoiding an obstacle, making lateral movements while maintaining a vertical orientation, performing circular movements; it can also deal with human-body kinematics or complex trees with multiple ramifications. The paper [Efficient motor learning through action-perception cycles in deep kinematic inference](https://link.springer.com/chapter/10.1007/978-3-031-47958-8_5) extends this model with an algorithms that allows to learn the kinematic chain during goal-directed behavior.

Check [this](https://priorelli.github.io/blog/) and [this](https://priorelli.github.io/projects/) for additional guides and projects.

## HowTo

### Start the simulation

The simulation can be launched through *main.py*, either with the option `-m` for manual control, `-s` for the IE (shallow) model, `-d` for the deep hierarchical model (deep), `-j` for the standard Jacobian control, or `-a` for choosing the parameters from the console. If no option is specified, the last one will be launched. For the manual control simulation, the arm can be moved with the keys `Z`, `X`, `A`, `S`, `LEFT`, `RIGHT`, `UP` and `DOWN`.

Plots can be generated through *plot.py*, either with the option `-d` for the belief trajectories, `-s` for the scores, or `-v` for generating a video of the simulation.

The folder *reference/video/* contains a few videos about the applications described in the paper.

### Advanced configuration

More advanced parameters can be manually set from *config.py*. Custom log names are set with the variable `log_name`. The number of trials and steps can be set with the variables `n_trials` and `n_steps`, respectively.

The parameter `lr_length` controls the learning rate of the length beliefs. If it is set to 0, these beliefs will not be updated.

Different precisions have been used for the shallow and deep models: for the former, the visual precision `pi_vis` has been set to 1.0, while the extrinsic precision `pi_ext` to 0.05; for the latter, the visual precision `pi_vis` has been set to 0.1, while the extrinsic precision `pi_ext` to 0.5.

The variable `task` affects the goal of the active inference agent, and can assume the following values:
1. `reach`: an intention to reach a red target is set at the last level. This corresponds to the reach and track tasks;
2. `avoid`: an intention to avoid a green obstacle is set at every level. This corresponds to the avoid task;
3. `both`: the first two conditions are active simultaneously. This corresponds to the reach and avoid, and track and avoid tasks;
4. `infer`: the proprioceptive precision `pi_prop` is set to 0, and a random arm configuration is set at every trial. This has been used to assess perceptual performances (e.g., in Figure 2B).

The variable `context` specifies whether (`dynamic`) or not (`static`) the red target and the green obstacle move. The velocity is set by `target_vel`.

The arm configuration is defined through the dictionary `joints`. The value `link` specifies the joint to which the new one is attached; `angle` encodes the starting value of the joint; `limit` defines the min and max angle limits.

If needed, target and obstacle positions and directions (encoded in the arrays `obj_pos` and `obj_dirs`) can be manually through the function `sample_objects` in *environment/window.py*.

### Active inference

The script *simulation/inference.py* contains a subclass of `Window` in *environment/window.py*, which is in turn a subclass `pyglet.window.Window`. The only overriden function is `update`, which defines the instructions to run in a single cycle. Specifically, the subclass `Inference` initializes the agent and the objects; during each update, it retrieves proprioceptive and visual observations through functions defined in *environment/window.py*, calls the function `inference_step` of the agent, and finally moves the arm and the objects.

There are three different classes for the agents, corresponding to the shallow, deep, or jacobian models. All of them have a similar function `inference_step`.

Useful trajectories computed during the simulations are stored through the class `Log` in *environment/log.py*.

Note that all the variables are normalized between -1 and 1 to ensure that every contribution to the belief updates has the same magnitude.

## Required libraries

matplotlib==3.6.2

numpy==1.24.1

pandas==1.5.3

pyglet==1.5.23

seaborn==0.12.2
