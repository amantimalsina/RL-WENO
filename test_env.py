from env import BurgersEnv

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = BurgersEnv()
    rewards = []
    for i in range(200):
        env.render()
        trivial_action = [3 / 10, 3 / 5, 1 / 10]
        action = np.array([trivial_action for i in range(400 + 3)])
        _, reward, _, _ = env.step(action)
        rewards.append(reward)

    plt.figure(figsize=(8, 5))
    plt.plot(rewards, 'bo-', markersize=3)
    plt.grid(axis='y')
    plt.show()
