from env import BurgersEnv

import numpy as np


if __name__ == '__main__':
    env = BurgersEnv()
    rewards = []

    s, _ = env.reset()
    done = False

    while not done:
        env.render()
        trivial_action = [3 / 10, 3 / 5, 1 / 10]
        a = np.array([trivial_action for i in range(400 + 3)])
        s_prime, r, done, info = env.step(a)

        s = s_prime
        done = done[0]

    print("Baseline reward: ", r.sum())
