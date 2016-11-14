import numpy as np

from scipy.stats import entropy
from scipy import optimize
from scipy import ndimage as ndi

from skimage import transform


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
    H, bin_edges = np.histogramdd([np.ravel(A), np.ravel(B)], bins=255)
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

    return I # * G


# Generalize this for N-d
def register(A, B):
    def build_tf(p):
        r, tx, ty = p
        return transform.SimilarityTransform(rotation=r,
                                             translation=(tx, ty))

    def cost(p):
        tf = build_tf(p)
        B_prime = transform.warp(B, tf, order=3)

        return -1 * alignment(A, B_prime, sigma=1.5)

    res = optimize.minimize(cost, [0, 0, 0], callback=lambda x: print('x->', x),
                            method='Powell')
    print('opt:', res.x)

    return build_tf(res.x)


if __name__ == "__main__":
    from skimage import data, transform, color

    img0 = transform.rescale(color.rgb2gray(data.astronaut()), 0.3)
    img1 = transform.rotate(img0, 30)

    tf = register(img0, img1)
    corrected = transform.warp(img1, tf)

    import matplotlib.pyplot as plt

    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(img0, cmap='gray')
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(corrected, cmap='gray')

    plt.show()
