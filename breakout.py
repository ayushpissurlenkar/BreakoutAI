import gym
from agents import Agents
import atari_py

class Breakout:

    def __init__(self, agents):
        self.agents = agents

    def runOpenAi(self):
        """ Given an agent simulate the agents in the openAI environment

            :return: simulation
        """
        env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type='grayscale')
        obv = env.reset()

        # Observation and Action Space
        obs_space = env.observation_space
        action_space = env.action_space



        print(type(obs_space), action_space)

        for _ in range(1000):
            env.step(env.action_space.sample()) # take a random action

        env.close()

if __name__ == '__main__':
    agent = Agents()
    game = Breakout(agent)
    game.runOpenAi()









