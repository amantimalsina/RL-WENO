from env import BurgersEnv

import numpy as np


if __name__ == '__main__':
    env = BurgersEnv()

    t = 0
    score = 0
    done = False
    s = env.reset()
    while t < 200:
        env.render()
        trivial_action = [3 / 10, 3 / 5, 1 / 10]
        a = np.array([trivial_action for i in range(400 + 3)])
        s_prime, r, dones, info = env.step(a)

        # Finish a timestep
        if not done:
            score += r.sum()

        t += 1
        s = s_prime
        done = dones[0]

    print("Baseline reward: ", score)
