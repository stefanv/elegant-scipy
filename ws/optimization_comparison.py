import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
#           'COBYLA',
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


def optimization_paths(axis):
    rosenbrock, rosenbrock_J, rosenbrock_H = rosenbrock_f(a=1, b=100)
    path = {}

    x, y = np.ogrid[-2:2:0.05, -1:3:0.05]
    ax.plot_surface(x, y, rosenbrock(x, y), rstride=1, cstride=1,
                    cmap='viridis', norm=LogNorm(), linewidth=0,
                    edgecolor='none', alpha=0.4)

    ax.set_xlim([-2, 2.0])
    ax.set_ylim([-1, 3.0])
    ax.set_zlim([0, 2500])
    ax.set_aspect('equal')

    x0 = (-0.5, 2.5)

    for method in methods:
        print('Optimizing with {}'.format(method))

        path = [x0]
        res = optimize.minimize(lambda p: rosenbrock(*p),
                                x0=x0,
                                jac=lambda p: rosenbrock_J(*p),
                                hess=lambda p: rosenbrock_H(*p),
                                method=method,
                                callback=lambda p: path.append(p))

        path = np.array(path)
        x, y = path.T
        ax.plot(x, y, rosenbrock(x, y), linewidth=3, label=method)
        ax.scatter(path[-1, 0], path[-1, 1] , res.fun, 'o',
                   s=200, depthshade=False)

    ax.legend()


if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig, azim=-128, elev=43)

    optimization_paths(axis=ax)

    plt.show()
