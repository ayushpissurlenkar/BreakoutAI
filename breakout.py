import sys
import gym
from agents import Agents
from qlearningAgents import QLearningAgents

class Breakout:

    def __init__(self, alpha=0.5, gamma=0.5, epsilon=0.3, numTraining=10000):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.numTraing = numTraining

    def runOpenAi(self):
        """ Given an agent simulate the agents in the openAI environment

            :return: simulation
        """
        env = gym.make('MountainCar-v0')

        # Initialize Agents
        agents = QLearningAgents(env, self.alpha, self.gamma, self.epsilon)

        for idx in range(self.numTraing):
            obv = env.reset()
            done = False

            while not done:
                action = agents.getAction(tuple(obv))
                new_obv, reward, done, info = env.step(action)
                agents.update(tuple(obv), action, tuple(new_obv), reward)
                obv = new_obv
                env.render()
                print(action, done, reward, idx)

        env.close()

if __name__ == '__main__':
    game = Breakout()
    game.runOpenAi()