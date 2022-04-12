import sys

import gym
from scipy.spatial import distance
import math
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class Environment:

    def __init__(self, env):
        self.env = env

    def get_discrete_state_breakout_v1(self, new_state):
        """ Pre-process the breakout game state representation

        :param state: represent the actual observation from the environment
        :return: represent the discrete state of the actual state
        """
        new_x = new_state[99]
        new_y = new_state[101]

        # Pruning Space above the block
        if new_y < 85:
            new_y = 0

        ball_position = (new_x, new_y)
        paddle_position = new_state[72]
        blocks = tuple(new_state[0:30])
        direction_vector = (new_state[103], new_state[105])
        vel = new_state[95]



        return tuple([ball_position, paddle_position, vel, direction_vector])

    def get_discrete_state_breakout_polar(self, prev, state):
        """
        Represent the Polar Coordinate Representation of paddle and ball

        :param state: Represent given current state
        :return: polor Representation of the breakout paddle and ball
        """
        old_x = int(prev[99])
        old_y = int(prev[101])
        new_x = int(state[99])
        new_y = int(state[101])
        paddle_position = int(state[72])

        rel_x = paddle_position - new_x
        paddle_y = -1 * 177 % 256
        rel_y = -1 * new_y % 256

        distance_to_ball = round(distance.euclidean((new_x, rel_y),(paddle_position, paddle_y)))

        angle = 90
        if rel_x != 0:
            angle = round(math.degrees(math.atan((rel_y - paddle_y)/rel_x)))

        ball_direction = (new_x - old_x, new_y - old_y)
        wall_detect = 36 <= new_y <= 76 or 180 <= new_y <= 220

        return tuple([angle, distance_to_ball, ball_direction, wall_detect])


    def get_discrete_state_mountaincart(self, state):
        """ Pre-process the mountain cart game state representation

        :param state: represent the actual observation from the environment
        :return: condensed state repsentation
        """
        buckets = (self.env.observation_space.high - self.env.observation_space.low) \
                  / [10, 10]
        # print(f'observations space {self.env.observation_space.high}, and {self.env.observation_space.low}')
        # print(f'print Bucket{buckets}')
        discrete_state = (state - self.env.observation_space.low) / buckets
        return tuple(discrete_state.astype(np.int))

    def breakout_reward_func(self, prev_obv, curr_obv, in_reward):
        prev_live = prev_obv[57]
        curr_live = curr_obv[57]

        # If Agent aren't able to bounce the ball back
        if curr_live < prev_live:
            # Return Reward of Negative 1
            return -1


        # Only Reward when paddle hit the ball back
        ball_y = curr_obv[101]
        new_directional_v = curr_obv[103]
        old_directional_v = prev_obv[103]

        if 170 <= ball_y < 207 and new_directional_v != old_directional_v:
            return 1

        return 0