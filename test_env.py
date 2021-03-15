import numpy as np
from env import BurgersEnv


if __name__ == '__main__':
    env = BurgersEnv()
    for i in range(200):
        env.render()
        trivial_action = [3 / 10, 3 / 5, 1 / 10]
        action = np.array([trivial_action for i in range(400 + 3)])
        env.step(action)
