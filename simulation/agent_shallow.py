import numpy as np
from numpy.linalg import norm
import utils
import config as c


# Define agent class
class AgentShallow:
    def __init__(self, arm):
        self.kinematics = arm.kinematics
        self.inverse = arm.inverse
        self.limits = utils.normalize(arm.limits, c.norm_polar)

        # Initialize belief and action
        self.mu_int = np.zeros((c.n_orders, c.n_joints, 2))
        self.mu_ext = np.zeros((c.n_orders, 2))

        self.a = np.zeros(c.n_joints)

    def g_ext(self):
        """
        Get extrinsic belief
        """
        mu_int_denorm = self.mu_int[0].copy()
        mu_int_denorm[:, 0] = utils.denormalize(mu_int_denorm[:, 0],
                                                c.norm_polar)

        new_mu_ext = self.kinematics(*mu_int_denorm.T)[-1, :2]

        return new_mu_ext

    def grad_ext(self, E_ext):
        """
        Get extrinsic gradient
        :param E_ext: extrinsic prediction error
        """
        mu_int_denorm = self.mu_int[0].copy()
        mu_int_denorm[:, 0] = utils.denormalize(mu_int_denorm[:, 0],
                                                c.norm_polar)

        grad_int = np.zeros_like(self.mu_int[0])
        for j in range(c.n_joints):
            inv = self.inverse(np.sum(mu_int_denorm[:j + 1, 0]),
                               mu_int_denorm[j, 1])[:2]
            for n in range(j + 1):
                grad_int[n] += inv

        # Gradient of normalization
        grad_int *= (c.norm_polar[1] - c.norm_polar[0])
        grad_int *= np.pi / 180

        lkh_int = np.c_[E_ext.dot(grad_int.T), np.zeros(c.n_joints)]

        return lkh_int

    def get_p(self):
        """
        Get predictions
        """
        p_ext = self.g_ext()
        p_prop = self.mu_int[0, :, 0].copy()
        p_vis = self.mu_ext[0].copy()

        return p_ext, p_prop, p_vis

    def get_i(self, target_joint, target_pos):
        """
        Get intentions
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        """
        joint_norm = utils.normalize(target_joint, c.norm_polar)
        pos_norm = utils.normalize(target_pos, c.norm_cart)

        i_int = self.mu_int[0].copy()
        i_ext = self.mu_ext[0].copy()

        # Only first joint
        # i_int[0, 0] = joint_norm[0]

        # Every joint
        # i_int[:, 0] = joint_norm

        # Final position
        i_ext = pos_norm[-1]

        return i_int, i_ext

    def get_e_g(self, S, P):
        """
        Get generative prediction errors
        :param S: observations
        :param P: predictions
        """
        E_g = [s - p for s, p in zip(S, P)]

        return E_g

    def get_e_mu(self, I):
        """
        Get dynamics prediction errors
        :param I: intentions
        """
        E_i = [(I[0] - self.mu_int[0]) * c.k_int,
               (I[1] - self.mu_ext[0]) * c.k_ext]

        return self.mu_int[1] - E_i[0], self.mu_ext[1] - E_i[1]

    def get_likelihood(self, E_g):
        """
        Get likelihood components
        :param E_g: generative prediction errors
        """
        lkh = {}

        lkh['int'] = self.grad_ext(E_g[0]) * c.pi_ext
        lkh['prop'] = np.zeros_like(self.mu_int[0])

        # For testing inference
        if c.task in ['reach', 'both']:
            lkh['prop'][:, 0] = E_g[1] * c.pi_prop

        lkh['vis'] = E_g[2] * c.pi_vis

        lkh['forward_ext'] = -E_g[0] * c.pi_ext

        return lkh

    def get_mu_dot(self, lkh, E_mu, obstacle_pos):
        """
        Get belief update
        :param lkh: likelihood components
        :param E_mu: dynamics prediction errors
        :param obstacle_pos: obstacle position
        """
        mu_int_dot = np.zeros_like(self.mu_int)
        mu_ext_dot = np.zeros_like(self.mu_ext)

        # Update likelihoods
        mu_int_dot[0] = self.mu_int[1] + lkh['prop'] + lkh['int']
        mu_ext_dot[0] = self.mu_ext[1] + lkh['vis'] + lkh['forward_ext']

        # Intentions
        if c.task in ['reach', 'both']:
            mu_int_dot[1] -= E_mu[0]
            mu_ext_dot[1] -= E_mu[1]

            # Circular trajectory
            # angle = np.arctan2(self.mu_ext[1, 1],
            #                    self.mu_ext[1, 0]) + 0.003
            # new_vel = np.cos(angle), np.sin(angle)
            # e_mu = self.mu_ext[1] - np.array(new_vel) * c.k_ext
            # self.mu_ext_dot[1] -= e_mu

        # Avoid obstacles
        if c.task in ['avoid', 'both']:
            avoid_o = self.get_rep_force(obstacle_pos)
            mu_ext_dot[1] -= avoid_o

        return mu_int_dot, mu_ext_dot

    def get_a_dot(self, e_prop):
        """
        Get action update
        :param e_prop: proprioceptive error
        """
        return -c.dt * e_prop

    def integrate(self, mu_int_dot, mu_ext_dot, a_dot):
        """
        Integrate with gradient descent
        :param mu_int_dot: intrinsic belief update
        :param mu_ext_dot: extrinsic belief update
        :param a_dot: action update
        """
        # Update belief
        self.mu_int[0] += c.dt * mu_int_dot[0] * c.gain_int
        if c.task != 'infer':
            self.mu_int[0, :, 0] = np.clip(self.mu_int[0, :, 0],
                                           *self.limits.T)
        self.mu_int[1] += c.dt * mu_int_dot[1]

        self.mu_ext[0] += c.dt * mu_ext_dot[0] * c.gain_ext
        self.mu_ext[1] += c.dt * mu_ext_dot[1]

        # Update action
        self.a += c.dt * a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    def init_belief(self, angles):
        """
        Initialize belief
        :param angles: initial arm joint angles
        """
        self.mu_int[0, :, 0] = utils.normalize(angles, c.norm_polar)
        self.mu_int[0, :, 1] = utils.normalize(c.lengths, c.norm_cart)
        self.mu_ext[0] = self.kinematics(angles, self.mu_int[0, :, 1])[-1, :2]

        # Linear trajectory
        # self.mu_ext[1] = np.array([0.0, 0.1])

    def get_rep_force(self, obstacle_pos):
        """
        Compute repulsive force
        :param obstacle_pos: obstacle position
        """
        pos_norm = utils.normalize(obstacle_pos, c.norm_polar)

        avoid_dist = c.target_size + 20
        q_star = utils.normalize(avoid_dist, c.norm_polar)
        error_r = pos_norm - self.mu_ext[0]
        error_r_norm = norm(error_r)

        if error_r_norm > q_star:
            rep_force = np.zeros(2)
        else:
            rep_force = c.k_rep * (1 / q_star - 1 / error_r_norm) \
                        * (1 / error_r_norm ** 2) * (error_r / error_r_norm)

        return self.mu_ext[1] - rep_force

    def inference_step(self, S, target_joint, target_pos, obstacle_pos):
        """
        Run an inference step
        :param S: observations
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        :param obstacle_pos: obstacle position
        """
        # Get predictions
        P = self.get_p()

        # Get intentions
        I = self.get_i(target_joint, target_pos)

        # Get generative prediction errors
        E_g = self.get_e_g((self.mu_ext[0], S[0], S[1][-1]), P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get likelihood components
        likelihood = self.get_likelihood(E_g)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, E_mu, obstacle_pos[-1])

        # Get action update
        a_dot = self.get_a_dot(E_g[1] * c.pi_prop)

        # Update
        self.integrate(*mu_dot, a_dot)

        return utils.denormalize(self.a, c.norm_polar) * c.gain_a, P
