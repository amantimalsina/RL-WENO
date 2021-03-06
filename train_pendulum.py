from RL.envs import NormalizedEnv
from RL.utils import plot_result
from RL.agent import DDPGAgent
from RL.models import Pi, Q

import gym
import pickle
from argparse import ArgumentParser


def main(args):
    env = NormalizedEnv(gym.make('Pendulum-v0'))
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    pi = Pi(
        dim_state=dim_state,
        dim_hidden1=args.dim_hidden1,
        dim_hidden2=args.dim_hidden2,
        dim_action=dim_action
    )
    q = Q(
        dim_state=dim_state,
        dim_hidden1=args.dim_hidden1,
        dim_hidden2=args.dim_hidden2,
        dim_action=dim_action
    )
    agent = DDPGAgent(
        env,
        pi,
        q,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        lr_pi=args.lr_pi,
        lr_q=args.lr_q,
        warmup_step=args.warmup_step,
        render=args.render
    )

    scores = 0.0
    history = []
    for i in range(args.n_episodes):
        scores += agent.run_episode()
        if (i+1) % args.print_interval == 0:
            print(f"[Episode {i+1}] Avg Score: {scores / args.print_interval}")
            history.append(scores / args.print_interval)
            scores = 0.0

    with open("./ddpg.pkl", 'wb') as f:
        pickle.dump(history, f)

    plot_result(history, args.print_interval)


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dim_hidden1', type=int, default=400)
    parser.add_argument('--dim_hidden2', type=int, default=300)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_step', type=int, default=2000)
    parser.add_argument('--lr_pi', type=float, default=0.0001)
    parser.add_argument('--lr_q', type=float, default=0.001)
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--print_interval', type=int, default=10)
    parser.add_argument('--render', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_arguments()
    main(args)
