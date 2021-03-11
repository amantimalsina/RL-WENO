from .utils import interpolate

import numpy as np
import matplotlib.pyplot as plt


class BurgersEnv:
    def __init__(self):
        self.N = 400
        self.T = 0.4
        self.C = 0.4  # Constant for choosing stable 'dt'
        self.x = np.linspace(-1, 1, self.N + 1)
        self.dx = 2 / self.N
        self.dt = self.C * self.dx

        self.reset()  # Initialize state, vb_plus, vb_minus

    def reset(self):
        """
        Set "self.state" to the initial condition. i.e. -1 * sin(\pi x).

        (Future work)
        Some randomness will be added here. We can consider initial conditions like
        - lambda * sin(\pi x) + (1-lambda) * cos(\pi x) where lambda is in [0, 1].
        - k * sin(\pi(x - a)) + b where k is a scale constant, a and b is translation constants.
        """
        self.state = -np.sin(np.pi * self.x)
        self._reset_vb()

        return self.state

    def _reset_vb(self):
        """
        (Future work)
        Where do the operations applied to "vb[3:N+4]" come from ?
        """
        N = self.N

        self.vb_plus = np.zeros(
            shape=len(self.x) + 6)  # +6 indicates three ghost cells on the left and on the right right.
        self.vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 + self.state)
        self.vb_plus[:3] = self.vb_plus[3]
        self.vb_plus[N + 4:] = self.vb_plus[N + 3]

        self.vb_minus = np.zeros(shape=len(self.x) + 6)
        self.vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 - self.state)
        self.vb_minus[:3] = self.vb_minus[3]
        self.vb_minus[N + 4:] = self.vb_minus[N + 3]

    def step(self):
        """
        action: np.ndarray with shape the (??, 3) such that each row is weights for interpolation.

        (Future work)
        Take an action and return (next state, reward, done, info)
        """
        N = self.N
        y0 = self.state

        # RK1
        vm, _ = interpolate(self.vb_plus, (3 / 10, 3 / 5, 1 / 10), N)
        _, vp = interpolate(self.vb_minus, (3 / 10, 3 / 5, 1 / 10), N)

        fminus = vm[1:N + 2] + vp[2:N + 3]
        fplus = vm[0:N + 1] + vp[1:N + 2]

        flux = -(fminus - fplus) / self.dx
        y1 = y0 + self.dt * flux
        self.state = y1
        self.vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 + self.state)
        self.vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 - self.state)

        # RK2
        vm, _ = interpolate(self.vb_plus, (3 / 10, 3 / 5, 1 / 10), N)
        _, vp = interpolate(self.vb_minus, (3 / 10, 3 / 5, 1 / 10), N)

        fminus = vm[1:N + 2] + vp[2:N + 3]
        fplus = vm[0:N + 1] + vp[1:N + 2]

        flux = -(fminus - fplus) / self.dx
        y1 = 3 / 4 * y0 + 1 / 4 * (self.state + self.dt * flux)
        self.state = y1
        self.vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 + self.state)
        self.vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 - self.state)

        # RK 3
        vm, _ = interpolate(self.vb_plus, (3 / 10, 3 / 5, 1 / 10), N)
        _, vp = interpolate(self.vb_minus, (3 / 10, 3 / 5, 1 / 10), N)

        fminus = vm[1:N + 2] + vp[2:N + 3]
        fplus = vm[0:N + 1] + vp[1:N + 2]

        flux = -(fminus - fplus) / self.dx
        y1 = 1 / 3 * y0 + 2 / 3 * (self.state + self.dt * flux)
        self.state = y1
        self.vb_plus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 + self.state)
        self.vb_minus[3:N + 4] = 1 / 2 * (1 / 2 * self.state ** 2 - self.state)

        return self.state

    def render(self):
        plt.cla()
        plt.grid()
        plt.xlim(-1, 1)
        plt.ylim(-1.2, 1.2)
        plt.plot(self.x, self.state, 'o-b', markersize=3, fillstyle='none', linewidth=1, alpha=0.4)
        plt.pause(0.0001)
