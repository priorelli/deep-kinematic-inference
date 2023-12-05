import utils
import config as c
import time
from simulation.manual_control import ManualControl
from simulation.inference import Inference


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.manual_control:
        c.fps = 200
        sim = ManualControl()

    elif options.shallow:
        sim = Inference('shallow')

    elif options.jacobian:
        sim = Inference('jacobian')

    elif options.deep:
        c.pi_vis = 0.1
        c.pi_ext = 0.5
        sim = Inference('deep')

    else:
        print('Choose model:')
        print('0 --> shallow')
        print('1 --> deep')
        print('2 --> jacobian')
        model = input('Model: ')

        print('Choose task:')
        print('0 --> reach target')
        print('1 --> avoid obstacle')
        print('2 --> reach and avoid')
        print('3 --> infer configuration')
        task = input('Task: ')

        print('\nChoose context:')
        print('0 --> static environment')
        print('1 --> dynamic environment')
        context = input('Context: ')

        model = 'shallow' if model == '0' else 'deep' if model == '1' \
            else 'jacobian'
        if model == 'deep':
            c.pi_vis = 0.1
            c.pi_ext = 0.5

        c.task = 'reach' if task == '0' else 'avoid' if task == '1' \
            else 'both' if task == '2' else 'infer'
        if task in ['1', '2']:
            c.n_steps = 8000

        c.context = 'static' if context == '0' else 'dynamic'

        time.sleep(0.5)
        sim = Inference(model)

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
