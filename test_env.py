from env import BurgersEnv


if __name__ == '__main__':
    env = BurgersEnv()
    for i in range(200):
        env.render()
        env.step()
