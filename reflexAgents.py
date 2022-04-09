import random
import util

from agents import Agents


class ReflexAgents(Agents):
    """
    Simple Reflex Agent that bounced the ball back

    """

    def __init__(self):
        self.env = None
        self.episode = util.Counter()

    def getAction(self, state):
        """
        The Agent will recieve the Breakout game state representation where the agents
        must return an action depending on the model

        :param state:
        :return: action that will manipulate the environment
        """

        ball_pos, paddle_pos, vel, directional_v = state

        ball_pos_x, ball_pos_y = ball_pos

        if ball_pos_x > paddle_pos:
            # Moving to the Right
            return random.choice([2,1])
        elif ball_pos_x < paddle_pos:
            # Moving to the Left
            return random.choice([3,1])
        else:
            return random.choice([0,1])


    def loadEnvironment(self, env):
        """
        Given OpenAI Gym Environment load the environment to the agents

        :param env: Represent the OpenAI gym environment
        :return:
        """
        self.env = env

    def getEpisode(self):
        """ Return the latest episode that is saved

        :return: the epsidoe that is saved
        """
        return 0

    def exportQTable(self, episode):
        """ Export the trained model for later reference

        :param episode: represent the episode which is saved
        :return: pickle file of q-value table
        """

    def importQTable(self, filepath):
        """ Given the model set up the Q-table for the q-learning
            agent

        :param filepath: represent the file path of q-learning model
        :return: set the q-learning agent with loaded model
        """

    def update(self, state, action, newState, reward):
        """
        Update the Q-Table

        :param state:
        :param action:
        :param newState:
        :param reward:
        :return: Return a new Q-Value table
        """

