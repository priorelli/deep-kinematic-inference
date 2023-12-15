import numpy as np
from numpy.linalg import norm
import utils
import config as c


# Define agent class
class AgentJacobian:
    def __init__(self, arm):
        self.kinematics = arm.kinematics
        self.inverse = arm.inverse
        self.limits = utils.normalize(arm.limits, c.norm_polar)

        # Initialize belief and action
        self.mu_int = np.zeros((c.n_orders, c.n_joints, 2))

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

    def grad_ext(self, E_ext, pinv=False):
        """
        Get extrinsic gradient
        :param E_ext: extrinsic prediction error
        :param pinv: use pseudoinverse
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

        if pinv:
            pinv_int = np.linalg.pinv(grad_int.T)
            lkh_int = np.c_[pinv_int.dot(E_ext), np.zeros(c.n_joints)]
        else:
            lkh_int = np.c_[E_ext.dot(grad_int.T), np.zeros(c.n_joints)]

        return lkh_int

    def get_p(self):
        """
        Get predictions
        """
        p_prop = self.mu_int[0, :, 0].copy()
        p_vis = self.g_ext()

        return p_prop, p_vis

    def get_i(self, target_joint, target_pos):
        """
        Get intentions
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        """
        joint_norm = utils.normalize(target_joint, c.norm_polar)
        pos_norm = utils.normalize(target_pos, c.norm_cart)
        phi = np.sum(joint_norm)

        i_int = self.mu_int[0].copy()

        # Only first joint
        # i_int[0, 0] = joint_norm[0]

        # Every joint
        # i_int[:, 0] = joint_norm

        return i_int

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
        E_i = (I[0] - self.mu_int[0]) * c.k_int

        return self.mu_int[1] - E_i[0]

    def get_likelihood(self, E_g):
        """
        Get likelihood components
        :param E_g: sensory prediction errors
        """
        lkh = {}

        lkh['prop'] = np.zeros_like(self.mu_int[0])

        # For testing inference
        if c.task in ['reach', 'both']:
            lkh['prop'][:, 0] = E_g[0] * c.pi_prop

        lkh['vis'] = self.grad_ext(E_g[1]) * c.pi_ext

        return lkh

    def get_mu_dot(self, lkh, E_mu, target_pos):
        """
        Get belief update
        :param lkh: likelihood components
        :param E_mu: dynamics prediction errors
        :param target_pos: desired link positions
        """
        mu_int_dot = np.zeros_like(self.mu_int)

        # Update likelihoods
        mu_int_dot[0] = self.mu_int[1] + lkh['vis'] + lkh['prop']

        # Intentions
        if c.task in ['reach', 'both']:
            mu_int_dot[1] -= E_mu.copy()

        # Reach
        pos_norm = utils.normalize(target_pos, c.norm_cart)
        attractor = self.g_ext() - pos_norm[-1]

        # Transpose
        mu_int_dot[1] -= self.grad_ext(attractor) * 0.025

        # Pseudoinverse
        # mu_int_dot[1] -= self.grad_ext(attractor, pinv=True) * 0.3

        return mu_int_dot

    def get_a_dot(self, e_prop):
        """
        Get action update
        :param e_prop: proprioceptive error
        """
        return -c.dt * e_prop

    def integrate(self, mu_int_dot, a_dot):
        """
        Integrate with gradient descent
        :param mu_int_dot: belief update
        :param a_dot: action update
        """
        # Update belief
        self.mu_int[0] += c.dt * mu_int_dot[0] * c.gain_int
        if c.task != 'infer':
            self.mu_int[0, :, 0] = np.clip(self.mu_int[0, :, 0],
                                           *self.limits.T)
        self.mu_int[1] += c.dt * mu_int_dot[1]

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

    def inference_step(self, S, target_joint, target_pos, obstacle_pos):
        """
        Run an inference step
        :param S: observations
        :param target_joint: desired joints angles
        :param target_pos: desired link positions
        :param obstacle_pos: obstacle position
        """
        # Get predictions
        P = self.get_p()

        # Get intentions
        I = self.get_i(target_joint, target_pos)

        # Get sensory prediction errors
        E_g = self.get_e_g((S[0], S[1][-1]), P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get likelihood components
        likelihood = self.get_likelihood(E_g)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, E_mu, target_pos)

        # Get action update
        a_dot = self.get_a_dot(E_g[0] * c.pi_prop)

        # Update
        self.integrate(mu_dot, a_dot)

        return utils.denormalize(self.a, c.norm_polar) * c.gain_a, P
