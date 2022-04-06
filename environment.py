import numpy as np


class Environment:

    def __init__(self, env):
        self.env = env

    def get_discrete_state_breakout_v1(self, old_state, new_state):
        """ Pre-process the breakout game state representation

        :param state: represent the actual observation from the environment
        :return: represent the discrete state of the actual state
        """

        old_x = int(old_state[99])
        old_y = int(old_state[101])
        new_x = int(new_state[99])
        new_y = int(new_state[101])
        ball_position = (new_x, new_y)
        paddle_position = int(new_state[72])
        blocks = tuple(new_state[0:30])
        direction_vector = (new_x - old_x, new_y - old_y)

        return tuple([ball_position, paddle_position, direction_vector])

    def get_discrete_state_mountaincart(self, state):
        """ Pre-process the mountain cart game state representation

        :param state: represent the actual observation from the environment
        :return: condensed state repsentation
        """
        buckets = (self.env.observation_space.high - self.env.observation_space.low) \
                  / [10, 10]
        discrete_state = (state - self.env.observation_space.low) / buckets
        return tuple(discrete_state.astype(np.int))
