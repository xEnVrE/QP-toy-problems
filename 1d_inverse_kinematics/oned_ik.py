import matplotlib
import matplotlib.pyplot as plt
import numpy
import time
from oned_manip import Manipulator
from oned_manip_sim import ManipulatorSim
from ik_qp import IK
from matplotlib import rc


def main():

    # Manipulator parameters
    length = 1.0
    q_max = 70.0
    q_min = 10.0
    q_dot_max = 10.0

    # Solver parameters
    gain = 0.01
    q_des = 45.0 * numpy.pi / 180.0
    x_des = numpy.array([[numpy.cos(q_des)], [numpy.sin(q_des)]]) * length
    eps = 1e-3
    N_max = 1000

    # Sampling time
    dt = 1.0 / 100.0

    # Initial state
    q0 = 15.0 * numpy.pi / 180.0

    # Setup
    manip = Manipulator(length)
    sim = ManipulatorSim(q0, q_max, q_min, dt)
    ik = IK('proxqp', dt, gain)
    ik.set_limits(q_min * numpy.pi / 180.0, q_max * numpy.pi / 180.0)
    ik.set_max_abs_vel(q_dot_max * numpy.pi / 180.0)

    # Storage for plotting
    xs = []
    qs = []
    qdots = []
    ts = []

    # Solve until error < eps or # iters > N_max
    iters = 0
    x = manip.fk(sim.get_q())
    while (numpy.linalg.norm(x - x_des) > eps) and (iters < N_max):
        q = sim.get_q()
        x = manip.fk(q)
        J = manip.J(q)

        t0 = time.time()
        q_dot = ik.solve(q, x, x_des, J)
        exec_time = time.time() - t0

        sim.step(q_dot)

        xs.append(x)
        qs.append(q)
        qdots.append(q_dot)
        ts.append(exec_time)

        iters += 1

    xs = numpy.array(xs)
    qs = numpy.array(qs)
    qdots = numpy.array(qdots)
    ts = numpy.array(ts)

    # Plot results
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex = True)
    fig, ax = plt.subplots(1, 3, figsize = (18, 6))

    ax[0].plot(xs[:, 0, 0], xs[:, 1, 0], color = 'blue', label = '$\mathrm{Trajectory}$')
    ax[0].scatter(x_des[0], x_des[1], color = 'red', s = 100, label = '$\mathrm{Goal}$')
    ax[0].set_title('$\mathrm{State}$', fontsize = 20)
    ax[0].set_xlabel('$x\,\mathrm{(m)}$', fontsize = 20)
    ax[0].set_ylabel('$y\,\mathrm{(m)}$', fontsize = 20)
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].xaxis.set_tick_params(labelsize = 15)
    ax[0].yaxis.set_tick_params(labelsize = 15)
    ax[0].grid()
    ax[0].legend(fontsize = 15)

    ax[1].plot(numpy.ones(qs.shape) * q_min, '--', color = 'black', label = '$\mathrm{Joint\,limit}$')
    ax[1].plot(numpy.ones(qs.shape) * q_max, '--', color = 'black')
    ax[1].plot(qs * 180.0 / numpy.pi, color = 'blue')
    ax[1].set_title('$\mathrm{Joint\,position\,}q$', fontsize = 20)
    ax[1].set_xlabel('$\mathrm{steps}$', fontsize = 20)
    ax[1].set_ylabel('$\mathrm{(deg)}$', fontsize = 20)
    ax[1].set_ylim(0.0, 90.0)
    ax[1].xaxis.set_tick_params(labelsize = 15)
    ax[1].yaxis.set_tick_params(labelsize = 15)
    ax[1].grid()
    ax[1].legend(fontsize = 15)

    ax[2].plot(numpy.ones(qdots.shape) * q_dot_max, '--', color = 'black', label = '$\mathrm{Velocity\,limit}$')
    ax[2].plot(numpy.ones(qdots.shape) * -q_dot_max, '--', color = 'black')
    ax[2].plot(qdots * 180.0 / numpy.pi, color = 'blue')
    ax[2].set_title('$\mathrm{Joint\,velocity\,}\dot{q}$', fontsize = 20)
    ax[2].set_xlabel('$\mathrm{steps}$', fontsize = 20)
    ax[2].set_ylabel('$\mathrm{(deg/s)}$', fontsize = 20)
    ax[2].set_ylim(-15.0, 15.0)
    ax[2].xaxis.set_tick_params(labelsize = 15)
    ax[2].yaxis.set_tick_params(labelsize = 15)
    ax[2].grid()
    ax[2].legend(fontsize = 15)

    print('Mean step execution time: ' + str(numpy.mean(ts)))

    plt.show()


if __name__ == '__main__':
    main()
