import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def sphere(x_c, y_c, z_c, r, axis, color="red"):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = x_c + r * np.cos(u) * np.sin(v)
    y = y_c + r * np.sin(u) * np.sin(v)
    z = z_c + r * np.cos(v)
    ax.plot_surface(x, y, z, color=color)


methods = ['Nelder-Mead',
           'Powell',
           'CG',
           'BFGS',
           'Newton-CG',
           'L-BFGS-B',
           'TNC',
#           'COBYLA',   # does not support callbacks
           'SLSQP',
           'dogleg',
           'trust-ncg'
               ]


def rosenbrock_f(a, b):
    """Return the Rosenbrock function, Jacobian & Hessian.

    Parameters
    ----------
    a, b : float
        Parameters defining the surface.  Typical values are a=1, b=100.

    Notes
    -----
    The Rosenbrock function has a minimum of 0 at ``(a, a**2)``.

    """
    def f(x, y):
        return (a - x)**2 + b * (y - x**2) ** 2

    def J(x, y):
        return np.array([-2 * (a - x) - 4 * b * x * (y - x**2),
                         2 * b * (y - x ** 2)])

    def H(x, y):
        return np.array([[2, -4 * b * x],
                         [-4 * b * x, 2 * b]])

    return f, J, H


def optimization_paths():
    rosenbrock, rosenbrock_J, rosenbrock_H = rosenbrock_f(a=1, b=100)
    path = {}

    fig, axes = plt.subplots(4, 3)
    fig.delaxes(axes[0, 1])
    fig.suptitle('Comparison of Optimation Methods on Rosenbrock Function')

    x, y = np.ogrid[-2:2:0.05, -1:3:0.05]
    extent = (-2, 2, -1, 3)

    z = rosenbrock(x, y).T
    axes[0, 0].matshow(z, norm=LogNorm(), origin='lower', extent=extent)
    axes[0, 0].set_title('Cost Function')

    x0 = (-0.5, 2.5)

    for n, method in enumerate(methods):
        print('Optimizing with {}'.format(method))

        path = [x0]
        res = optimize.minimize(lambda p: rosenbrock(*p),
                                x0=x0,
                                jac=lambda p: rosenbrock_J(*p),
                                hess=lambda p: rosenbrock_H(*p),
                                method=method,
                                callback=lambda p: path.append(p))

        path = np.array(path)
        px, py = path.T

        ax = axes.flat[n + 2]

        ax.contour(z, extent=extent, norm=LogNorm())
        ax.plot(px, py, linewidth=3)
        ax.set_aspect('equal')
        ax.scatter(path[-1, 0], path[-1, 1])
        ax.set_title(method)

    ax.legend()


if __name__ == '__main__':
    optimization_paths()
    plt.show()
