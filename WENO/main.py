from WENO.weno3 import WENO3

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Initialize variables
    N = 400
    T = 0.4
    C = 0.4  # constant for choosing stable 'dt'

    x = np.linspace(-1, 1, N + 1)
    dx = 2 / N
    dt = C * dx
    time_step = T / dt

    ub = -np.sin(np.pi * x)  # Initial condition

    # Ghost cells and impose B.C.
    vb_plus = np.zeros(shape=len(ub) + 6)
    vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 + ub)
    vb_plus[:3] = vb_plus[3]
    vb_plus[N + 4:] = vb_plus[N + 3]

    vb_minus = np.zeros(shape=len(ub) + 6)
    vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 - ub)
    vb_minus[:3] = vb_minus[3]
    vb_minus[N + 4:] = vb_minus[N + 3]

    time = 0
    plt.figure(figsize=(8, 5))
    for i in range(0, int(time_step) + 1):
        y0 = ub

        # RK 1
        vm, _ = WENO3(vb_plus, N)
        _, vp = WENO3(vb_minus, N)

        fminus = vm[1:N + 2] + vp[2:N + 3]
        fplus = vm[0:N + 1] + vp[1:N + 2]

        flux = -(fminus - fplus) / dx
        y1 = y0 + dt * flux
        ub = y1
        vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 + ub)
        vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 - ub)

        # RK 2
        vm, _ = WENO3(vb_plus, N)
        _, vp = WENO3(vb_minus, N)

        fminus = vm[1:N + 2] + vp[2:N + 3]
        fplus = vm[0:N + 1] + vp[1:N + 2]

        flux = -(fminus - fplus) / dx
        y1 = 3 / 4 * y0 + 1 / 4 * (ub + dt * flux)
        ub = y1
        vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 + ub)
        vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 - ub)

        # RK 3
        vm, _ = WENO3(vb_plus, N)
        _, vp = WENO3(vb_minus, N)

        fminus = vm[1:N + 2] + vp[2:N + 3]
        fplus = vm[0:N + 1] + vp[1:N + 2]

        flux = -(fminus - fplus) / dx
        y1 = 1 / 3 * y0 + 2 / 3 * (ub + dt * flux)
        ub = y1
        vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 + ub)
        vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * ub ** 2 - ub)

        time += dt
        plt.cla()
        plt.grid()
        plt.xlim(-1, 1)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('$x$', fontsize=15, fontweight='bold')
        plt.ylabel(f'$u(x,{time:.3f})$', fontsize=15, fontweight='bold')
        plt.title('Burger\'s Equation with $u_0 = -\sin{\pi x}$', fontsize=15, fontweight='bold')
        plt.plot(x, ub, 'o-b', markersize=3, label=f'$t=${i}', fillstyle='none', linewidth=1, alpha=0.4)
        plt.pause(0.0001)
