import numpy as np
from numpy.linalg import norm
import utils
import config as c


# Define agent class
class AgentDeep:
    def __init__(self, arm):
        self.kinematics = arm.kinematics
        self.inverse = arm.inverse
        self.idxs = arm.idxs
        self.limits = utils.normalize(arm.limits, c.norm_polar)

        # Initialize belief and action
        self.mu_int = np.zeros((c.n_orders, c.n_joints, 2))
        self.mu_ext = np.zeros((c.n_orders, c.n_joints + 1, 3))

        self.a = np.zeros(c.n_joints)

    def g_ext(self):
        """
        Get extrinsic belief
        """
        mu_int_denorm = self.mu_int[0].copy()
        mu_int_denorm[:, 0] = utils.denormalize(mu_int_denorm[:, 0],
                                                c.norm_polar)

        mu_ext_denorm = self.mu_ext[0].copy()
        mu_ext_denorm[:, 2] = utils.denormalize(mu_ext_denorm[:, 2],
                                                c.norm_polar)

        new_mu_ext = self.kinematics(*mu_int_denorm.T, mu_ext_denorm)
        new_mu_ext[:, 2] = utils.normalize(new_mu_ext[:, 2], c.norm_polar)

        return new_mu_ext

    def grad_ext(self, E_ext, P_ext):
        """
        Get extrinsic gradient
        :param E_ext: extrinsic prediction error
        :param P_ext: extrinsic prediction
        """
        p_ext_denorm = P_ext.copy()
        p_ext_denorm[:, 2] = utils.denormalize(p_ext_denorm[:, 2],
                                               c.norm_polar)

        grad_theta = np.array([self.inverse(phi, length) for phi, length in
                               zip(p_ext_denorm[1:, 2], self.mu_int[0, :, 1])])
        grad_length = np.array([[np.cos(np.radians(phi)),
                                 np.sin(np.radians(phi)), 0] for phi in
                                p_ext_denorm[1:, 2]])
        grad_length *= c.lr_length

        # Gradient of normalization
        grad_theta[:, :2] *= np.pi / 180
        grad_theta[:, :2] *= (c.norm_polar[1] - c.norm_polar[0])

        grad_ext = np.array([[[1, 0, 0], [0, 1, 0], grad.copy()]
                             for grad in grad_theta])

        lkh_int = np.c_[np.sum(E_ext[1:] * grad_theta, axis=1),
                        np.sum(E_ext[1:] * grad_length, axis=1)]

        lkh_ext = np.zeros_like(self.mu_ext[0])
        for j in range(1, c.n_joints):
            idx = self.idxs[j] + 1
            lkh_ext[idx] += grad_ext[idx].dot(E_ext[j + 1])

        return lkh_int * c.pi_ext, lkh_ext * c.pi_ext

    def get_p(self):
        """
        Get predictions
        """
        p_ext = self.g_ext()
        p_prop = self.mu_int[0, :, 0].copy()
        p_vis = self.mu_ext[0, :, :2].copy()

        return p_ext, p_prop, p_vis

    def get_i(self, target_joint, target_pos):
        """
        Get intentions
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        """
        joint_norm = utils.normalize(target_joint, c.norm_polar)
        pos_norm = utils.normalize(target_pos, c.norm_cart)
        phi = utils.normalize(90, c.norm_polar)

        i_int = self.mu_int[0].copy()
        i_ext = self.mu_ext[0].copy()

        # Only second joint
        # i_int[1, 0] = utils.normalize(100, c.norm_polar)

        # Every joint
        # i_int[:, 0] = joint_norm

        # Only final position
        i_ext[-1, :2] = pos_norm[-1]

        # Every position
        # i_ext[:, :2] = pos_norm

        # Mixed
        # i_int[0, 0] = joint_norm[0]
        # i_ext[-1, :2] = pos_norm[-1]

        # Only final orientation
        # i_ext[-1, 2] = phi

        # Final position and orientation
        # i_ext[-1] = [*pos_norm[-1], phi]

        # Point
        # vector = pos_norm[-1] - self.mu_ext[0, -1, :2]
        # angle = np.degrees(np.arctan2(vector[1], vector[0]))
        # phi = utils.normalize((angle + 360) % 360, c.norm_polar)
        # i_ext[-1, 2] = phi

        return i_int, i_ext

    def get_e_g(self, S, P):
        """
        Get sensory prediction errors
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
        # k_ext = [c.k_ext / c.gain_phi, c.k_ext / c.gain_phi, c.k_ext]

        E_i = [(I[0] - self.mu_int[0]) * c.k_int,
               (I[1] - self.mu_ext[0]) * c.k_ext]

        return self.mu_int[1] - E_i[0], self.mu_ext[1] - E_i[1]

    def get_likelihood(self, E_g, P):
        """
        Get likelihood components
        :param E_g: sensory prediction errors
        :param P: predictions
        """
        lkh = {}

        lkh['int'], lkh['ext'] = self.grad_ext(E_g[0], P[0])
        lkh['prop'] = np.zeros_like(self.mu_int[0])

        # For testing inference
        if c.task != 'infer':
            lkh['prop'][:, 0] = E_g[1] * c.pi_prop

        lkh['vis'] = np.c_[E_g[2] * c.pi_vis, np.zeros(c.n_joints + 1)]

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
        mu_ext_dot[0] = self.mu_ext[1] + lkh['ext'] + lkh['vis'] + \
                            lkh['forward_ext']

        # Intentions
        if c.task in ['reach', 'both']:
            mu_int_dot[1] -= E_mu[0]
            mu_ext_dot[1] -= E_mu[1]

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
        self.mu_int[0] += c.dt * mu_int_dot[0]
        if c.task != 'infer':
            self.mu_int[0, :, 0] = np.clip(self.mu_int[0, :, 0],
                                           *self.limits.T)
        self.mu_int[1] += c.dt * mu_int_dot[1]
        self.mu_int[0, :, 1] = np.clip(self.mu_int[0, :, 1], 0, 1)

        self.mu_ext[0] += c.dt * mu_ext_dot[0]
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

        # self.mu_int[0, :, 1] = 0.1
        # self.mu_int[0, :, 1] = np.random.rand(c.n_joints)

        self.mu_ext[0] = self.kinematics(angles, self.mu_int[0, :, 1])
        self.mu_ext[0, :, 2] = utils.normalize(self.mu_ext[0, :, 2],
                                               c.norm_polar)

    def get_rep_force(self, obstacle_pos):
        """
        Compute repulsive force
        :param obstacle_pos: obstacle position
        """
        pos_norm = utils.normalize(obstacle_pos, c.norm_cart)

        q_star = utils.normalize(c.avoid_dist, c.norm_cart)
        error_r = pos_norm - self.mu_ext[0, :, :2]
        error_r_norm = norm(error_r, axis=1)

        rep_force = np.zeros((c.n_joints + 1, 3))
        for j in range(1, c.n_joints + 1):
            if error_r_norm[j] < q_star:
                rep_force[j, :2] = c.k_rep * \
                                   (1 / q_star - 1 / error_r_norm[j]) * \
                                   (1 / error_r_norm[j] ** 2) *\
                                   (error_r[j] / error_r_norm[j])

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

        # Get sensory prediction errors
        E_g = self.get_e_g((self.mu_ext[0], *S), P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get likelihood components
        likelihood = self.get_likelihood(E_g, P)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, E_mu, obstacle_pos[-1])

        # Get action update
        a_dot = self.get_a_dot(E_g[1] * c.pi_prop)

        # Update
        self.integrate(*mu_dot, a_dot)

        return utils.denormalize(self.a, c.norm_polar) * c.gain_a, P
