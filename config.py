# Window
width = 400
height = 300
off_x = -70
off_y = -120
fps = 0

# Agent
dt = 0.3
a_max = 10.0
gain_a = 20.0
gain_phi = 10.0
k_rep = 1e-3
lr_length = 0.0
gain_int = 1.0  # 1.5
gain_ext = 1.0  # 0.1

pi_prop = 1.0
pi_vis = 1.0  # [1.0-s, 0.1-d]
pi_ext = 0.05  # [0.05-s, 0.5-d]

k_int = 0.1
k_ext = 1.0  # 0.01

w_p = 0  # 2e-3
w_a = 0  # 5e-5

# Inference
task = 'reach'  # reach, avoid, both, infer
context = 'static'  # static, dynamic
log_name = ''

target_size = 15
target_vel = 0.1
reach_dist = target_size / 2
avoid_dist = target_size * 1.7

n_trials = 20
n_steps = 500
n_orders = 2

# Arm
start = [0, 20, 40, 30, 30, -30, 0, 0]
lengths = [50, 70, 90, 20]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'limit': (-5, 10), 'size': (lengths[0], 40)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'limit': (-5, 130), 'size': (lengths[1], 30)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'limit': (-5, 130), 'size': (lengths[2], 26)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'limit': (-90, 90), 'size': (lengths[3], 26)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]
