import sys
import gym
from agents import Agents
from qlearningAgents import QLearningAgents
import numpy as np

class Breakout:

    def __init__(self, agent, numTraining=100000, show_every_ep=1000):
        self.agent = agent
        self.numTraining = numTraining
        self.showEveryEp = show_every_ep


    def runOpenAi(self):
        """ Given an agent simulate the agents in the openAI environment

            :return: simulation
        """

        def calc_discrete_state(state):
            """ Pre-process the breakout game state representation

            :param state: represent the actual observation from the environment
            :return: represent the discrete state of the actual state
            """
            ball_position = (state[99], state[101])
            paddle_position = state[72]
            blocks = tuple(state[0:30])

            return tuple([ball_position, paddle_position, blocks])

        for idx in range(self.numTraining):
            if idx % self.showEveryEp == 0:
                env = gym.make('Breakout-ram-v4', obs_type='ram', render_mode="human")
                self.agent.loadEnvironment(env)
            else:
                env = gym.make('Breakout-ram-v4', obs_type='ram', render_mode=None)
                self.agent.loadEnvironment(env)

            obv = env.reset()
            done = False

            dis_obv = calc_discrete_state(obv)

            while not done:
                # Retrieve action from agent
                action = self.agent.getAction(dis_obv)

                # Get new Observation from the environment
                new_obv, reward, done, info = env.step(action)
                dis_new_obv = calc_discrete_state(new_obv)

                # Update Agent
                self.agent.update(dis_obv, action, dis_new_obv, reward)
                dis_obv = dis_new_obv

                print(action, done, reward, idx)

            if idx % self.showEveryEp == 0:
                self.agent.exportQTable(idx)

        env.close()

if __name__ == '__main__':
    qLearningAgent = QLearningAgents(alpha=0.05, gamma=0.8, epsilon=0.2)
    game = Breakout(qLearningAgent)
    game.runOpenAi()