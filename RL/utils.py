import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque


class ReplayBuffer:
    def __init__(self, max_buffer_size=50000):
        self.buffer = deque(maxlen=max_buffer_size)

    def push(self, transition):
        """
        transition: a tuple of five elements (s, a, r, s_prime, done)
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Note that the current size of the buffer must be greater than "batch_size". This condition can be satisfied
        by storing transitions without training in the early stage of training, which is called "warmup"
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_prime, done = map(np.float32, zip(*batch))

        s = np.vstack(s)
        s_prime = np.vstack(s_prime)
        a = np.vstack(a)
        # Convert row vectors to columns vectors
        r = r.reshape((-1, 1))
        done = done.reshape((-1, 1))

        return s, a, r, s_prime, done

    def size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, theta=0.15, dt=0.01, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.dt = dt
        self.sigma = sigma
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x


def plot_result(history, interval=1):
    """
    reference: https://tykimos.github.io/2017/07/09/Training_Monitoring/
    """
    history = np.array(history)
    x = interval * np.arange(0, len(history))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, history, 'b-', label='Reward')
    ax.set_xlabel('Episode', fontsize=15)
    ax.set_ylabel('Reward', fontsize=15)
    ax.set_title('The average rewards of last 100 episodes', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend()
    plt.show()
