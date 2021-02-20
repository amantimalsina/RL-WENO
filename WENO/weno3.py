import numpy as np


def WENO3(vb, N):
    """
    WENO-JS scheme with k=3

    references
    ----------
    [Jingyang Guo and Jae-Hun Jung, 2016](https://arxiv.org/abs/1602.00183)
    """
    d0 = 3 / 10
    d1 = 3 / 5
    d2 = 1 / 10
    d0_tilda = d2
    d1_tilda = d1
    d2_tilda = d0
    epsilon = 1e-6

    vm = np.zeros(shape=N + 3)
    vp = np.zeros(shape=N + 3)

    for i in range(N + 3):
        vm0 = (1 / 3) * vb[i + 2] + (5 / 6) * vb[i + 3] + (-1 / 6) * vb[i + 4]
        vm1 = (-1 / 6) * vb[i + 1] + (5 / 6) * vb[i + 2] + (1 / 3) * vb[i + 3]
        vm2 = (1 / 3) * vb[i] + (-7 / 6) * vb[i + 1] + (11 / 6) * vb[i + 2]

        vp0 = (11 / 6) * vb[i + 2] + (-7 / 6) * vb[i + 3] + (1 / 3) * vb[i + 4]
        vp1 = (1 / 3) * vb[i + 1] + (5 / 6) * vb[i + 2] + (-1 / 6) * vb[i + 3]
        vp2 = (-1 / 6) * vb[i] + (5 / 6) * vb[i + 1] + (1 / 3) * vb[i + 2]

        beta0 = 13 / 12 * (vb[i + 2] - 2 * vb[i + 3] + vb[i + 4]) ** 2 + 1 / 4 * (
                    3 * vb[i + 2] - 4 * vb[i + 3] + vb[i + 4]) ** 2
        beta1 = 13 / 12 * (vb[i + 1] - 2 * vb[i + 2] + vb[i + 3]) ** 2 + 1 / 4 * (vb[i + 1] - vb[i + 3]) ** 2
        beta2 = 13 / 12 * (vb[i] - 2 * vb[i + 1] + vb[i + 2]) ** 2 + 1 / 4 * (
                    vb[i] - 4 * vb[i + 1] + 3 * vb[i + 2]) ** 2

        alpha0 = d0 / (beta0 + epsilon) ** 2
        alpha1 = d1 / (beta1 + epsilon) ** 2
        alpha2 = d2 / (beta2 + epsilon) ** 2
        alpha = alpha0 + alpha1 + alpha2

        alpha0_tilda = d0_tilda / (beta0 + epsilon) ** 2
        alpha1_tilda = d1_tilda / (beta1 + epsilon) ** 2
        alpha2_tilda = d2_tilda / (beta2 + epsilon) ** 2
        alpha_tilda = alpha0_tilda + alpha1_tilda + alpha2_tilda

        w0 = alpha0 / alpha
        w1 = alpha1 / alpha
        w2 = alpha2 / alpha

        w0_tilda = alpha0_tilda / alpha_tilda
        w1_tilda = alpha1_tilda / alpha_tilda
        w2_tilda = alpha2_tilda / alpha_tilda

        vm[i] = w0 * vm0 + w1 * vm1 + w2 * vm2
        vp[i] = w0_tilda * vp0 + w1_tilda * vp1 + w2_tilda * vp2

    return vm, vp
