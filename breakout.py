import sys
import gym
from agents import Agents
from ale_py import ALEInterface

class Breakout:

    def __init__(self, agents, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining=1000):
        self.agents = agents
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.numTraing = numTraining

    def runOpenAi(self):
        """ Given an agent simulate the agents in the openAI environment

            :return: simulation
        """
        env = gym.make('Breakout-ram-v4', render_mode='human', obs_type='ram')
        obv = env.reset()

        for _ in range(1000):

            obv, reward, done, info = env.step(3) # take a random action
            print(obv, reward, done, info)

        env.close()

if __name__ == '__main__':
    agent = Agents()
    game = Breakout(agent)
    game.runOpenAi()