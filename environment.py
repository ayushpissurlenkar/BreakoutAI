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

    def get_discrete_state_breakout_v2(self, old_state, new_state):
        """ Pre-process the breakout game state representation

        :param state: represent the actual observation from the environment
        :return: represent the relative position of paddle and ball
        """

        old_x = int(old_state[99])
        old_y = int(old_state[101])
        new_x = int(new_state[99])
        new_y = int(new_state[101])
        paddle_position = int(new_state[72])

        # Compute Direction of The Ball
        ball_is_moving = 'STAY'

        if new_x > old_x:
            ball_is_moving = 'MOVING_RIGHT'
        else:
            ball_is_moving = 'MOVING_LEFT'

        # Compute Ball position Relative to paddle position
        paddle_rel_ball = 'STAY'

        if new_x > paddle_position:
            paddle_rel_ball = 'BALL_IS_RIGHT_TO_PADDLE'
        else:
            paddle_rel_ball = 'BALL_IS_LEFT_TO_PADDLE'

        return tuple([ball_is_moving, paddle_rel_ball])

    def get_discrete_state_mountaincart(self, state):
        """ Pre-process the mountain cart game state representation

        :param state: represent the actual observation from the environment
        :return: condensed state repsentation
        """
        buckets = (self.env.observation_space.high - self.env.observation_space.low) \
                  / [10, 10]
        discrete_state = (state - self.env.observation_space.low) / buckets
        return tuple(discrete_state.astype(np.int))


    def breakout_reward_func(self, prev_obv, curr_obv, in_reward):
        prev_live = prev_obv[57]
        curr_live = curr_obv[57]

        # If Agent aren't able to bounce the ball back
        if curr_live < prev_live:
            # Return Reward of Negative 1
            return -1


        # If Paddle hit the ball
        ball_y = curr_obv[101]
        new_directional_v = curr_obv[103]
        old_directional_v = prev_obv[103]

        if 170 <= ball_y < 207 and new_directional_v != old_directional_v:
            return 1

        return in_reward
