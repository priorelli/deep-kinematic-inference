import numpy as np
import config as c
import utils


# Define log class
class Log:
    def __init__(self, model):
        self.model = model

        # Initialize logs
        self.actions = np.zeros((c.n_trials, c.n_steps, c.n_joints))

        self.angles = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.pos = np.zeros((c.n_trials, c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros_like(self.pos)

        self.phi = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.est_phi = np.zeros_like(self.phi)

        self.est_lengths = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.est_forward = np.zeros((c.n_trials, c.n_steps, c.n_joints + 1, 2))

        self.target_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.obstacle_pos = np.zeros((c.n_trials, c.n_steps, 2))

        self.success = np.zeros(c.n_trials)

    # Track logs for each time step
    def track(self, step, trial, agent, arm, target_pos, obstacle_pos, P):
        self.actions[trial, step] = agent.a

        self.angles[trial, step] = arm.angles
        if self.model == 'jacobian':
            est_angles = utils.denormalize(P[0], c.norm_polar)
        else:
            est_angles = utils.denormalize(P[1], c.norm_polar)
        self.est_angles[trial, step] = est_angles

        self.pos[trial, step] = arm.poses[:, :2]
        if self.model == 'shallow':
            est_pos = utils.denormalize(P[2], c.norm_cart)
            self.est_pos[trial, step, -1] = est_pos
        elif self.model == 'deep':
            est_pos = utils.denormalize(P[2], c.norm_cart)
            self.est_pos[trial, step] = est_pos
        elif self.model == 'jacobian':
            est_pos = utils.denormalize(P[1], c.norm_cart)
            self.est_pos[trial, step, -1] = est_pos
        else:
            est_pos = utils.denormalize(P[3], c.norm_cart)
            self.est_pos[trial, step] = est_pos

        self.phi[trial, step] = arm.poses[1:, 2]
        if self.model == 'deep':
            est_phi = utils.denormalize(agent.mu_ext[0, 1:, 2], c.norm_polar)
            self.est_phi[trial, step] = est_phi

        if self.model == 'deep':
            est_lengths = utils.denormalize(agent.mu_int[0, :, 1],
                                            c.norm_cart)
            self.est_lengths[trial, step] = est_lengths

        est_forward = arm.kinematics(est_angles)[:, :2]
        self.est_forward[trial, step] = est_forward

        self.target_pos[trial, step] = target_pos[-1]
        self.obstacle_pos[trial, step] = obstacle_pos[-1]

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_' + c.log_name,
                            actions=self.actions,
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos, est_pos=self.est_pos,
                            phi=self.phi, est_phi=self.est_phi,
                            est_lengths=self.est_lengths,
                            est_forward=self.est_forward,
                            target_pos=self.target_pos,
                            obstacle_pos=self.obstacle_pos,
                            success=self.success)
