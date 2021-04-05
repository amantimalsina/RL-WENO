from RL.models import Pi
from env import BurgersEnv

import torch


DIM_HIDDEN1 = 400
DIM_HIDDEN2 = 300
DIR_WEIGHT = './outputs/0331_flux/best_model_pi.bin'


if __name__ == '__main__':
    env = BurgersEnv()
    pi = Pi(DIM_HIDDEN1, DIM_HIDDEN2)

    state_dict = torch.load(DIR_WEIGHT)
    pi.load_state_dict(state_dict)

    with torch.no_grad():
        t = 0
        score = 0
        done = False
        s = env.reset()
        while t < 200:
            env.render()
            a = pi(torch.tensor(s, dtype=torch.float))
            a = a.detach().numpy()
            s_prime, r, dones, info = env.step(a)

            # Finish a timestep
            if not done:
                score += r.sum()
            t += 1
            s = s_prime
            done = dones[0]

    print("Model's reward: ", score)
