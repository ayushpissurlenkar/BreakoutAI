import sys
import gym
from agents import Agents
from qlearningAgents import QLearningAgents
import numpy as np

class Breakout:

    def __init__(self, alpha=0.05, gamma=0.8, epsilon=0.2, numTraining=100000):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.numTraining = numTraining


    def runOpenAi(self):
        """ Given an agent simulate the agents in the openAI environment

            :return: simulation
        """
        env = gym.make('Breakout-ram-v4', obs_type='ram', render_mode='human')

        # Initialize Agents
        agents = QLearningAgents(env, self.alpha, self.gamma, self.epsilon)

        def calc_discrete_state(state):
            """ Pre-process the breakout game state representation

            :param state: represent the actual observation from the environment
            :return: represent the discrete state of the actual state
            """
            """
            print(f'Observation space high {env.observation_space.high} \n'
                  f'Observation space low {env.observation_space.low} ')
    
            bucket = (env.observation_space.high - env.observation_space.low) \
                      /[10,10]
            discrete_state = (state - env.observation_space.low)/bucket
            print(f'Observation space high {env.observation_space.high} \n'
                  f'Observation space low {env.observation_space.low} \n'
                  f'Observation state space {state} \n'
                  f'Observation discrete state space {discrete_state.astype(np.int)}')
            """

            return tuple(state)

        for idx in range(self.numTraining):
            obv = env.reset()
            done = False

            dis_obv = calc_discrete_state(obv)

            while not done:
                action = agents.getAction(dis_obv)
                new_obv, reward, done, info = env.step(action)
                print(dis_obv)
                dis_new_obv = calc_discrete_state(new_obv)
                agents.update(dis_obv, action, dis_new_obv, reward)
                dis_obv = dis_new_obv
                print(action, done, reward, idx)

                """
                if idx % 1000 == 0:
                    env.render() 
                """

            if idx % 1000 == 0:
                agents.exportQTable(idx)

        env.close()

if __name__ == '__main__':
    game = Breakout()
    game.runOpenAi()