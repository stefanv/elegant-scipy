"""

Implementation of:

 Pluim et al., Image registration by maximization of combined mutual
 information and gradient information, IEEE Transactions on Medical
 Imaging, 19(8) 2000

 Pluim et al., Mutual-Information-Based Registration of Medical
 Images: A Survey, IEEE Transactions on Medical Imaging, 22(8) 2003

"""

import numpy as np

from scipy.stats import entropy
from scipy import optimize
from scipy import ndimage as ndi

from skimage import transform
from skimage.util import random_noise


def gaussian_pyramid(image, levels=7):
    """Make a Gaussian image pyramid.

    Parameters
    ----------
    image : array of float
        The input image.
    max_layer : int, optional
        The number of levels in the pyramid.

    Returns
    -------
    pyramid : list of array of float
        A list of Gaussian pyramid levels, starting with the top
        (lowest resolution) level.
    """
    pyramid = [image]
    rows, cols = image.shape[0], image.shape[1]

    for level in range(levels):
        rows = np.ceil(rows / 2)
        cols = np.ceil(cols / 2)
        blurred = ndi.gaussian_filter(image, sigma=2/3)
        image = transform.resize(blurred, (rows, cols))
        pyramid.append(image)

    return pyramid[::-1]


def mutual_information(A, B, normalized=True):
    """Compute the normalized mutual information.

    The normalized mutual information is given by:

                H(A) + H(B)
      Y(A, B) = -----------
                  H(A, B)

    where H(X) is the entropy ``- sum(x log x) for x in X``.

    Parameters
    ----------
    A, B : ndarray
        Images to be registered.
    normalized : bool
        Whether or not to normalize the mutual information.
    """
    # TODO: Check if bin edges need to be specified
    H, bin_edges = np.histogramdd([np.ravel(A), np.ravel(B)], bins=100)
    H /= np.sum(H)

    H_A = entropy(np.sum(H, axis=0))
    H_B = entropy(np.sum(H, axis=1))
    H_AB = entropy(np.ravel(H))

    if normalized:
        return (H_A + H_B) / H_AB
    else:
        return H_A + H_B - H_AB


def gradient(image, sigma=1):
    gaussian_filtered = ndi.gaussian_filter(image, sigma=sigma,
                                            mode='constant', cval=0)
    return np.gradient(gaussian_filtered)


def gradient_norm(g):
    return np.linalg.norm(g, axis=-1)


def gradient_similarity(A, B, sigma=1, scale=True):
    """For each pixel, calculate the angle between the gradients of A & B.

    Parameters
    ----------
    A, B : ndarray
        Images.
    sigma : float
        Sigma for the Gaussian filter, used to calculate the image gradient.

    Notes
    -----
    In multi-modal images, gradients may often be similar but point
    in opposite directions.  This weighting function compensates for
    that by mapping both 0 and pi to 1.

    Different imaging modalities can highlight different structures.  We
    are only interested in edges that occur in both images, so we scale the
    similarity by the minimum of the two gradients.

    """
    g_A = np.dstack(gradient(A, sigma=sigma))
    g_B = np.dstack(gradient(B, sigma=sigma))

    mag_g_A = gradient_norm(g_A)
    mag_g_B = gradient_norm(g_B)

    alpha = np.arccos(np.sum(g_A * g_B, axis=-1) /
                        (mag_g_A * mag_g_B))

    w = (np.cos(2 * alpha) + 1) / 2

    w[np.isclose(mag_g_A, 0)] = 0
    w[np.isclose(mag_g_B, 0)] = 0

    return w * np.minimum(mag_g_A, mag_g_B)


def alignment(A, B, sigma=1.5, normalized=True):
    I = mutual_information(A, B, normalized=normalized)
    G = np.sum(gradient_similarity(A, B, sigma=sigma))

    return I * G
#    return I


def build_tf(p):
    rotation, shear, tx, ty = p
    return transform.AffineTransform(rotation=rotation,
                                     shear=shear,
                                     translation=(tx, ty))


def cost(p, X, Y):
    tf = build_tf(p)
    Y_prime = transform.warp(Y, tf, output_shape=X.shape, order=3)

    return -1 * alignment(X, Y_prime)
    #return np.sum((X - Y_prime)**2) / np.prod(X.shape)


# Generalize this for N-d
def register(A, B):
    #pyramid_A = tuple(transform.pyramid_gaussian(A, downscale=2))
    #pyramid_B = tuple(transform.pyramid_gaussian(B, downscale=2))
    pyramid_A = gaussian_pyramid(A, levels=10)
    pyramid_B = gaussian_pyramid(B, levels=10)

    N = range(len(pyramid_A) - 1, -1, -1)
    image_pairs = zip(N, pyramid_A, pyramid_B)

    p = np.array([0, 0, 0, 0])

    for (n, X, Y) in image_pairs:
        if X.shape[0] < 5:
            continue

        print('   .  ')
        print('  / \ Pyramid scaled down by 2x {}'.format(n))
        print(' /   \ ')
        print('.-----.')

        p[1:] *= 2

        res = optimize.minimize(cost,
                                p,
                                args=(X, Y),
                                method='Powell')
        p = res.x

        print('Angle:', np.rad2deg(res.x[0]))
        print('Offset:', res.x[1:] * 2 ** n)
        print('Cost function:', res.fun)
        print('')

    return build_tf(p)


if __name__ == "__main__":
    from skimage import data, transform, color, io, img_as_float
    import matplotlib.pyplot as plt

    dataset = 'prokudin-gorskii'
    # dataset = 'astronaut'

    explore_rotation = False

    if dataset == 'astronaut':
        img0 = transform.rescale(color.rgb2gray(data.astronaut()), 0.4)

        img0 = transform.rescale(color.rgb2gray(data.astronaut()), 0.3)

        theta = 60
        img1 = transform.rotate(img0, theta)
        img1 = random_noise(img1, mode='gaussian', seed=0, mean=0, var=1e-3)

        tf = register(img0, img1)
        corrected = transform.warp(img1, tf, order=3)

    elif dataset == 'prokudin-gorskii':
#        scaling = '_small'
        scaling = ''

        img_r = color.rgb2gray(img_as_float(io.imread('../images/prokudin_gorskii_00998_r{}.png'.format(scaling))))
        img_g = color.rgb2gray(img_as_float(io.imread('../images/prokudin_gorskii_00998_g{}.png'.format(scaling))))
        img_b = color.rgb2gray(img_as_float(io.imread('../images/prokudin_gorskii_00998_b{}.png'.format(scaling))))

        final_shape = img_g.shape + (3,)
        out = np.zeros(final_shape)

        tf = register(img_g, img_r)
        corrected_r = transform.warp(img_r, tf, output_shape=img_g.shape, order=3)

        tf = register(img_g, img_b)
        corrected_b = transform.warp(img_b, tf, output_shape=img_g.shape, order=3)

        out[..., 0] = corrected_r
        out[..., 1] = img_g
        out[..., 2] = corrected_b

        print('Writing output to /tmp/registered.png')
        plt.imsave('/tmp/registered.png', out)

    if explore_rotation:

        f, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(img0, cmap='gray')
        ax0.set_title('Input image')
        ax1.imshow(img1, cmap='gray')
        ax1.set_title('Transformed image + noise')
        ax2.imshow(corrected, cmap='gray')
        ax2.set_title('Registered image')

        print('Calculating cost function profile...')
        f, ax0 = plt.subplots()
        angles = np.linspace(-theta - 20, -theta + 20, 51)
        costs = [-1 * alignment(img0, transform.rotate(img1, angle)) for angle in angles]
        ax0.plot(angles, costs)
        ax0.set_title('Cost function around angle of interest')
        ax0.set_xlabel('Angle')
        ax0.set_ylabel('Cost')

        plt.show()
