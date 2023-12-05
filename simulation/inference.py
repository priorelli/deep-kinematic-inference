import time
import utils
import config as c
from environment.window import Window
from environment.log import Log


# Define inference class
class Inference(Window):
    def __init__(self, model):
        super().__init__()
        # Initialize agent
        if model == 'shallow':
            from simulation.agent_shallow import AgentShallow as Agent
        elif model == 'jacobian':
            from simulation.agent_jacobian import AgentJacobian as Agent
        else:
            from simulation.agent_deep import AgentDeep as Agent

        self.agent = Agent(self.arm)

        # Initialize error tracking
        self.log = Log(model)

        # Initialize trial
        self.reset_trial()
        self.time = time.time()

    def update(self, dt):
        # Get observations
        S = self.get_joint_obs(), self.get_visual_obs()

        # Perform free energy step
        action, P = self.agent.inference_step(
            S, self.obj_joints[0], self.obj_pos[0], self.obj_pos[1])

        # Update arm
        if c.task != 'infer':
            action_noise = utils.add_gaussian_noise(action, c.w_a)
            self.arm.update(action_noise)

        # Move objects
        if c.context == 'dynamic':
            self.move_objects()

        # Track log
        self.log.track(self.step, self.trial - 1, self.agent,
                       self.arm, self.obj_pos[0], self.obj_pos[1], P)

        # Print info
        self.step += 1
        if self.step % 50 == 0:
            utils.print_info(self.trial, self.success, self.step)
            # utils.print_inference(self.trial - 1, self.step - 1,
            #                       self.log, self.arm)

        # Reset trial
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        self.success += self.task_done()

        # Simulation done
        if self.trial == c.n_trials:
            self.log.success = self.success
            utils.print_score(self.log, time.time() - self.time)
            self.log.save_log()
            self.stop()
        # Initialize simulation
        else:
            # Sample objects
            self.sample_objects()

            # Initialize belief
            if self.trial == 0:
                self.agent.init_belief(self.arm.angles)

            # Set different configuration for inference
            if c.task == 'infer':
                self.arm.set_rotation(self.obj_joints[0])

            self.step = 0
            self.trial += 1
