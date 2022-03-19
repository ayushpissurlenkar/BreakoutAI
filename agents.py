import util

class Agents:
    """
    An agent must define a get Action method which will return correspond action
    """

    def __init__(self):
        pass

    def getAction(self, state):
        """
        The Agent will recieve the Breakout game state representation where the agents
        must return an action depending on the model

        :param state:
        :return: action that will manipulate the environment
        """

        util.raiseNotDefined()

    def loadEnvironment(self, env):
        """
        Given OpenAI Gym Environment load the environment to the agents

        :param env: Represent the OpenAI gym environment
        :return:
        """