from .utils import interpolate

import numpy as np
import matplotlib.pyplot as plt


class BurgersEnv:
    """
    state : the flux vector of a given u
    action: a set of weights calculated from consecutive 5 points
    reward: a set of the TV norm and the maximum value of temporal difference of consecutive 5 points
    info  : the solution vector u
    """
    def __init__(self):
        self.N = 400
        self.T = 0.4
        self.C = 0.4  # Constant for choosing stable 'dt'
        self.x = np.linspace(-1, 1, self.N + 1)
        self.dx = 2 / self.N
        self.dt = self.C * self.dx

        self.t = 0
        self.u = None
        self.state = None
        self.reset()

    @staticmethod
    def _calculate_reward(u, u_prev, t):
        """
        Calculate reward
        """
        N = len(u) - 1
        # TV norm for each 5 points
        ub = np.zeros(len(u) + 7)
        ub[3:N + 4] = u
        tv_vector = np.abs(ub[1:] - ub[:-1])  # vector whose each elements is u_{i+1} - u_{i}
        tv_matrix = np.stack([tv_vector[i:i + 5] for i in range(N + 3)])  # Stack all consecutive 5 points
        tv_norm = np.sum(tv_matrix, axis=1)  # Summation of each row

        # Maximum temporal difference for each 5 points
        ub_prev = np.zeros(len(u) + 6)
        ub_prev[3:N + 4] = u_prev
        td_vector = np.abs(ub[:-1] - ub_prev)  # vector whose each elements is u_{i}^{t} - u_{i}^{t-1}
        td_matrix = np.stack([td_vector[i:i + 5] for i in range(N + 3)])  # Stack all consecutive 5 points
        td_maximum = np.max(td_matrix, axis=1)  # Apply maximum to each row

        # Total reward
        reward = -1 * (tv_norm + 10 * td_maximum) + np.repeat(t, N + 3)

        return reward

    def reset(self):
        """
        Set "self.u" to the initial condition, i.e. -1 * sin(\pi x)
        Note that an initial state is the flux vector of self.u, not self.u

        (Future work)
        Some randomness will be added here. We can consider initial conditions like
        - lambda * sin(\pi x) + (1-lambda) * cos(\pi x) where lambda is in [0, 1].
        - k * sin(\pi(x - a)) + b where k is a scale constant, a and b is translation constants.
        """
        N = self.N
        self.u = -np.sin(np.pi * self.x)
        self.state = np.zeros(len(self.x) + 6)
        self.state[3:N + 4] = self.u
        self.state[:3] = self.state[3]
        self.state[N + 4:] = self.state[N + 3]

        self.flux = np.zeros(len(self.x) + 6)  # +6 indicates three ghost cells on the left and on the right right.
        self.flux[3:N + 4] = 1 / 2 * self.u ** 2
        self.flux[:3] = self.flux[3]
        self.flux[N + 4:] = self.flux[N + 3]

        self.t = 0
        state = np.stack([self.state[i:i + 5] for i in range(N + 3)])

        return state

    def step(self, action):
        """
        action: np.ndarray with shape the (N+3, 3) such that each row is weights for interpolation.

        (Future work)
        Define reward function.
        """
        N = self.N
        u_prev = self.u.copy()
        y0 = self.u

        # Euler Method
        vm, vp = interpolate(self.flux, action, N)
        fminus = vm[1:N + 2]
        fplus = vp[1:N + 2]
        flux = -(fminus - fplus) / self.dx
        y1 = y0 + self.dt * flux
        self.u = y1
        self.state[3:N + 4] = self.u
        self.flux[3:N + 4] = 1 / 2 * self.u ** 2

        # Calculate reward
        reward = self._calculate_reward(self.u, u_prev, self.t)

        # Set next state
        state = np.stack([self.state[i:i + 5] for i in range(N + 3)])

        # Verify episode
        self.t += self.dt
        done = (self.t >= self.T) or (np.any(np.abs(self.u) > 1.05))
        done = np.repeat(done, N + 3)

        return state, reward, done, None

    def render(self):
        plt.cla()
        plt.grid()
        plt.xlim(-1, 1)
        plt.ylim(-1.2, 1.2)
        plt.plot(self.x, self.u, 'o-b', markersize=3, fillstyle='none', linewidth=1, alpha=0.4)
        plt.pause(0.00001)


class BurgersFluxEnv:
    """
    state : the flux vector of a given u
    action: a set of weights calculated from consecutive 5 points
    reward: a set of the TV norm and the maximum value of temporal difference of consecutive 5 points
    info  : the solution vector u
    """
    def __init__(self):
        self.N = 400
        self.T = 0.4
        self.C = 0.4  # Constant for choosing stable 'dt'
        self.x = np.linspace(-1, 1, self.N + 1)
        self.dx = 2 / self.N
        self.dt = self.C * self.dx

        self.t = 0
        self.u = None
        self.state = None
        self.reset()

    @staticmethod
    def _calculate_reward(u, u_prev, t):
        """
        Calculate reward
        """
        N = len(u) - 1
        # TV norm for each 5 points
        ub = np.zeros(len(u) + 7)
        ub[3:N + 4] = u
        tv_vector = np.abs(ub[1:] - ub[:-1])  # vector whose each elements is u_{i+1} - u_{i}
        tv_matrix = np.stack([tv_vector[i:i + 5] for i in range(N + 3)])  # Stack all consecutive 5 points
        tv_norm = np.sum(tv_matrix, axis=1)  # Summation of each row

        # Maximum temporal difference for each 5 points
        ub_prev = np.zeros(len(u) + 6)
        ub_prev[3:N + 4] = u_prev
        td_vector = np.abs(ub[:-1] - ub_prev)  # vector whose each elements is u_{i}^{t} - u_{i}^{t-1}
        td_matrix = np.stack([td_vector[i:i + 5] for i in range(N + 3)])  # Stack all consecutive 5 points
        td_maximum = np.max(td_matrix, axis=1)  # Apply maximum to each row

        # Total reward
        reward = -1 * (tv_norm + 10 * td_maximum) + np.repeat(t, N + 3)

        return reward

    def reset(self):
        """
        Set "self.u" to the initial condition, i.e. -1 * sin(\pi x)
        Note that an initial state is the flux vector of self.u, not self.u

        (Future work)
        Some randomness will be added here. We can consider initial conditions like
        - lambda * sin(\pi x) + (1-lambda) * cos(\pi x) where lambda is in [0, 1].
        - k * sin(\pi(x - a)) + b where k is a scale constant, a and b is translation constants.
        """
        N = self.N
        self.u = -np.sin(np.pi * self.x)

        self.state = np.zeros(len(self.x) + 6)  # +6 indicates three ghost cells on the left and on the right right.
        self.state[3:N + 4] = 1 / 2 * self.u ** 2
        self.state[:3] = self.state[3]
        self.state[N + 4:] = self.state[N + 3]

        self.t = 0
        state = np.stack([self.state[i:i + 5] for i in range(N + 3)])

        return state

    def step(self, action):
        """
        action: np.ndarray with shape the (N+3, 3) such that each row is weights for interpolation.

        (Future work)
        Define reward function.
        """
        N = self.N
        u_prev = self.u.copy()
        y0 = self.u

        # Euler Method
        vm, vp = interpolate(self.state, action, N)
        fminus = vm[1:N + 2]
        fplus = vp[1:N + 2]
        flux = -(fminus - fplus) / self.dx
        y1 = y0 + self.dt * flux
        self.u = y1
        self.state[3:N + 4] = 1 / 2 * self.u ** 2

        # Calculate reward
        reward = self._calculate_reward(self.u, u_prev, self.t)

        # Set next state
        state = np.stack([self.state[i:i + 5] for i in range(N + 3)])

        # Verify episode
        self.t += self.dt
        done = (self.t >= self.T) or (np.any(np.abs(self.u) > 1.05))
        done = np.repeat(done, N + 3)

        return state, reward, done, None

    def render(self):
        plt.cla()
        plt.grid()
        plt.xlim(-1, 1)
        plt.ylim(-1.2, 1.2)
        plt.plot(self.x, self.u, 'o-b', markersize=3, fillstyle='none', linewidth=1, alpha=0.4)
        plt.pause(0.00001)
