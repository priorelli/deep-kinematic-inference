import numpy as np
import pyglet
from pyglet.shapes import Circle, Rectangle
import utils
import config as c
from environment.arm import Arm


# Define window class
class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Deep kinematic inference',
                         vsync=False)
        # Initialize arm
        self.arm = Arm()

        # Initialize target and obstacle
        self.obj_joints = np.zeros((2, c.n_joints))
        self.obj_pos = np.zeros((2, c.n_joints + 1, 2))
        self.obj_dirs = np.zeros((2, 2))

        # Initialize agent
        self.agent = None

        # Initialize simulation variables
        self.step, self.trial, self.success = 0, 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.offset = (c.width / 2 + c.off_x, c.height / 2 + c.off_y)

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        objects = self.draw_screen()
        self.batch.draw()
        self.fps_display.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    # Draw screen
    def draw_screen(self):
        objects = set()

        # Move coordinates on screen
        target_w = self.obj_pos[0, -1] + self.offset
        obstacle_w = self.obj_pos[1, -1] + self.offset
        pos_w = self.arm.poses[:, :2] + self.offset

        # Draw obstacle
        if c.task in ['reach', 'both']:
            objects.add(Circle(*target_w, c.target_size, segments=20,
                               color=(200, 100, 0), batch=self.batch))

        if c.task in ['avoid', 'both']:
            objects.add(Circle(*obstacle_w, c.target_size, segments=20,
                               color=(100, 200, 0), batch=self.batch))

        # Draw arm
        objects.add(Circle(*self.offset, 20, segments=20,
                           color=(0, 100, 200), batch=self.batch))

        for j in range(c.n_joints):
            objects = self.draw_arm(objects, j, pos_w[j + 1])

        return objects

    # Draw arm
    def draw_arm(self, objects, n, pos):
        length, width = self.arm.size[n]

        # Draw link
        link = Rectangle(*pos, length, width,
                         color=(0, 100, 200), batch=self.batch)
        link.anchor_position = (length, width / 2)

        link.rotation = -self.arm.poses[n + 1, 2]
        objects.add(link)

        # Draw joint
        objects.add(Circle(*pos, width / 2, segments=20,
                           color=(0, 100, 200), batch=self.batch))

        return objects

    # Get visual observation
    def get_visual_obs(self):
        return utils.normalize(self.arm.kinematics()[:, :2], c.norm_cart)

    # Get proprioceptive observation
    def get_joint_obs(self):
        angles_noise = utils.add_gaussian_noise(self.arm.angles, c.w_p)
        return utils.normalize(angles_noise, c.norm_polar)

    # Check if task is successful
    def task_done(self):
        return np.linalg.norm(self.obj_pos[0, -1] -
                              self.arm.poses[-1, :2]) < c.reach_dist

    # Generate target and obstacle randomly
    def sample_objects(self):
        for o in range(2):
            # Sample position
            self.obj_joints[o] = np.random.uniform(*self.arm.limits.T)
            self.obj_pos[o] = self.arm.kinematics(self.obj_joints[o])[:, :2]

            # Sample velocity
            angle = np.random.rand() * 2 * np.pi
            self.obj_dirs[o] = np.array((np.cos(angle), np.sin(angle)))

        # self.obj_pos[0, -1] = [56.6, 165.7]
        # self.obj_pos[1, -1] = [40, 140]
        # self.obj_joints[0] = [10, 40, 60, 0]
        # self.obj_dirs[1] = np.array([-0.5, 0.86])
        # self.obj_pos[0, -1] = [-30, 140]
        # self.obj_pos[0] = self.arm.kinematics(self.obj_joints[0])[:, :2]

    # Move objects
    def move_objects(self):
        for o in range(2):
            self.obj_pos[o, -1] += c.target_vel * self.obj_dirs[o]

            # Bounce
            pos_w = self.obj_pos[o, -1] + self.offset
            if not c.target_size < pos_w[0] < c.width - c.target_size:
                self.obj_dirs[o, 0] = -self.obj_dirs[o, 0]
            if not c.target_size < pos_w[1] < c.height - c.target_size:
                self.obj_dirs[o, 1] = -self.obj_dirs[o, 1]
