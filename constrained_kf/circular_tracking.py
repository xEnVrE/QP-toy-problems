import matplotlib
import matplotlib.pyplot as plt
import numpy
from circular_constraint import CircularConstraint
from kf_qp import KF
from motion_model import MotionModel
from measurement_model import MeasurementModel
from matplotlib import rc


def main():

    numpy.random.seed(0)

    dt = 1.0 / 100.0

    # Generate measurements of a particle constrained in circular motion
    noise_std = 0.1
    radius = 1.0

    f = 0.2
    T = 4.0
    steps = int(T / dt)
    times = numpy.array([i * dt for i in range(steps)])

    gt_x = numpy.cos(2 * numpy.pi * f * times) * radius
    gt_y = numpy.sin(2 * numpy.pi * f * times) * radius
    gt = numpy.stack((gt_x, gt_y))

    meas_x = gt_x + numpy.random.normal(0, noise_std, steps)
    meas_y = gt_y + numpy.random.normal(0, noise_std, steps)
    meas = numpy.stack((meas_x, meas_y))

    # Instantiate motion model
    psd = 1.0
    motion_model = MotionModel(dt, 2, psd)

    # Instantiate measurement model
    cov = 0.1
    measurement_model = MeasurementModel(2, cov)

    # Instantiate filter
    x_0 = numpy.array([[0.1], [0.1], [0.0], [0.0]])
    P_0 = numpy.eye(4) * 0.01
    filter = KF('proxqp')
    filter.set_initial_condition(x_0, P_0)
    filter.set_motion_model(motion_model.A(), motion_model.Q())
    filter.set_measurement_model(measurement_model.H(), measurement_model.R())

    # Filter measurements
    exec_times, estimates = filter.filter_batch(meas)
    print('Mean step execution time: ' + str(numpy.mean(exec_times)))

    # Instantiate circular constraint
    gain = 5.0
    constraint = CircularConstraint(gain, radius)
    filter.set_initial_condition(x_0, P_0)
    filter.set_equality_constraint(constraint)
    _, estimates_w_const = filter.filter_batch(meas)

    # Plot results
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex = True)
    fig, ax = plt.subplots(1, 3, figsize = (19, 7))

    plot_0, = ax[0].plot(meas[0, :], meas[1, :], color = 'lightblue')#, label = '$\mathrm{Measurement}\,z$')
    plot_1, = ax[0].plot(estimates[0, :], estimates[2, :], color = 'blue')#, label = '$\mathrm{Estimate}\,x$')
    plot_2, = ax[0].plot(gt[0, :], gt[1, :], color = 'red')#, label = '$\mathrm{Ground\,truth}$')
    ax[0].set_title('$\mathrm{State}$', fontsize = 20)
    ax[0].set_xlabel('$x\,\mathrm{(m)}$', fontsize = 20)
    ax[0].set_ylabel('$y\,\mathrm{(m)}$', fontsize = 20)
    ax[0].grid()

    ax[1].plot(meas[0, :], meas[1, :], color = 'lightblue')
    ax[1].plot(estimates_w_const[0, :], estimates_w_const[2, :], color = 'blue')
    ax[1].plot(gt[0, :], gt[1, :], color = 'red')
    ax[1].set_title('$\mathrm{State\,(with\,circular\,constraint)}$', fontsize = 20)
    ax[1].set_xlabel('$x\,\mathrm{(m)}$', fontsize = 20)
    ax[1].set_ylabel('$y\,\mathrm{(m)}$', fontsize = 20)
    ax[1].grid()

    legend = fig.legend([plot_0, plot_1, plot_2],\
                        ['$\mathrm{Measurement}\,z$', '$\mathrm{Estimate}\,x$', '$\mathrm{Ground\,truth}$'],\
                        loc = 'upper left', ncol = 3, frameon = False, fontsize = 15)

    rmse = numpy.linalg.norm(numpy.stack((estimates[0, :] - gt_x, estimates[2, :] - gt_y)), axis = 0)
    rmse_w_const = numpy.linalg.norm(numpy.stack((estimates_w_const[0, :] - gt_x, estimates_w_const[2, :] - gt_y)), axis = 0)
    ax[2].plot(rmse, color = 'gray', label = '$\mathrm{w/o\,constraint}$')
    ax[2].plot(rmse_w_const, color = 'black', label = '$\mathrm{w/\,constraint}$')
    ax[2].set_title('$\mathrm{Error}$', fontsize = 20)
    ax[2].set_xlabel('$\mathrm{steps}$', fontsize = 20)
    ax[2].set_ylabel('$\mathrm{(m)}$', fontsize = 20)
    ax[2].legend(fontsize = 15)

    plt.show()


if __name__ == '__main__':
    main()
