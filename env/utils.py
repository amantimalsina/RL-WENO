import numpy as np


def interpolate(vb, weights, N):
    """
    weights: a tuple with three elements each of which is the weight for a stensil.
    """
    vm = np.zeros(shape=N + 3)
    vp = np.zeros(shape=N + 3)

    for i in range(N + 3):
        vm0 = (1 / 3) * vb[i + 2] + (5 / 6) * vb[i + 3] + (-1 / 6) * vb[i + 4]
        vm1 = (-1 / 6) * vb[i + 1] + (5 / 6) * vb[i + 2] + (1 / 3) * vb[i + 3]
        vm2 = (1 / 3) * vb[i] + (-7 / 6) * vb[i + 1] + (11 / 6) * vb[i + 2]

        vp0 = (11 / 6) * vb[i + 2] + (-7 / 6) * vb[i + 3] + (1 / 3) * vb[i + 2]
        vp1 = (1 / 3) * vb[i + 1] + (5 / 6) * vb[i + 2] + (-1 / 6) * vb[i + 3]
        vp2 = (-1 / 6) * vb[i] + (5 / 6) * vb[i + 1] + (1 / 3) * vb[i + 2]

        vm[i] = weights[0] * vm0 + weights[1] * vm1 + weights[2] * vm2
        vp[i] = weights[2] * vp0 + weights[1] * vp1 + weights[0] * vp2

    return vm, vp