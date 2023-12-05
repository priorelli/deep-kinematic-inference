from pyglet.window import key
import utils
import config as c
from environment.window import Window


# Define manual control class
class ManualControl(Window):
    def __init__(self):
        super().__init__()
        # Initialize trial
        self.reset_trial()

    def update(self, dt):
        # Get action from user
        action = self.get_pressed()

        # Update arm
        self.arm.update(action)

        # Update objects
        if c.context == 'dynamic':
            self.move_objects()

        # Print info
        self.step += 1
        if self.step % 50 == 0:
            utils.print_info(self.trial, self.success, self.step)

        # Reset trial
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        # Simulation done
        if self.trial == c.n_trials:
            self.stop()
        else:
            self.success += self.task_done()

            # Sample objects
            self.sample_objects()

            self.step = 0
            self.trial += 1

    # Get action from user input
    def get_pressed(self):
        return [(key.Z in self.keys) - (key.X in self.keys),
                (key.LEFT in self.keys) - (key.RIGHT in self.keys),
                (key.UP in self.keys) - (key.DOWN in self.keys),
                (key.A in self.keys) - (key.S in self.keys)]
