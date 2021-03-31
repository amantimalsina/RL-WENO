from RL.models import Pi
from env import BurgersEnv

import torch


DIM_HIDDEN1 = 400
DIM_HIDDEN2 = 300
DIR_WEIGHT = './outputs/0330/best_model_pi.bin'


if __name__ == '__main__':
    env = BurgersEnv()
    pi = Pi(DIM_HIDDEN1, DIM_HIDDEN2)

    state_dict = torch.load(DIR_WEIGHT)
    pi.load_state_dict(state_dict)

    with torch.no_grad():
        s, _ = env.reset()
        done = False

        t = 0
        score = 0
        while t < 200:
            env.render()
            a = pi(torch.tensor(s, dtype=torch.float))
            a = a.detach().numpy()
            s_prime, r, done, info = env.step(a)

            score += r.sum()
            s = s_prime
            done = done[0]
            t += 1

    print("Model's reward: ", score)
