import gym


class NormalizedEnv(gym.ActionWrapper):
    """
    The code is from https://github.com/antocapp/paperspace-ddpg-tutorial/blob/master/ddpg-pendulum-250.ipynb
    """
    def action(self, action):
        """
        Convert a given normalized action value to the original action value.
        In the Pendulum-v0 problem, our policy outputs a value from -1 to 1
        but the action space of the problem is originally from -2 to 2
        """
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        """
        Convert a given original action value to the normalized action value.
        """
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2
        return act_k_inv * (action - act_b)
