import sys
import gym
from environment import Environment
import numpy as np
from agents import Agents
from qlearningAgents import QLearningAgents
from reflexAgents import ReflexAgents
import numpy as np

class MountainCart:

    def __init__(self, agent, numTraining=1000000, show_every_ep=1000):
        self.agent = agent
        self.numTraining = numTraining
        self.showEveryEp = show_every_ep
        self.avgScore = []


    def runOpenAi(self):
        """ Given an agent simulate the agents in the openAI environment

            :return: simulation
        """

        loadedTraining = self.agent.getEpisode()

        for idx in range(self.numTraining):
            episode = idx + loadedTraining
            env = gym.make('MountainCar-v0')
            self.agent.loadEnvironment(env)

            obv = env.reset()
            done = False
            preprocess_env = Environment(env)

            dis_obv = preprocess_env.get_discrete_state_mountaincart(obv)

            # print(f'The initial observation of the block {dis_obv}')

            while not done:
                # Retrieve action from agent
                action = self.agent.getAction(dis_obv)

                prev_obv = obv

                # Get new Observation from the environment
                new_obv, reward, done, info = env.step(action)
                dis_new_obv = preprocess_env.get_discrete_state_mountaincart(new_obv)

                # Update Agent
                self.agent.update(dis_obv, action, dis_new_obv, reward)
                dis_obv = dis_new_obv
                obv = new_obv

            if 'TimeLimit.truncated' not in info:
                self.avgScore.append(1)
            else:
                self.avgScore.append(-1)




            if idx % self.showEveryEp == 0:
                avg_score = self.calculateAvgScore()
                print(f'The average score {avg_score} for interval {episode - self.showEveryEp} to {episode}')
                self.resetAvgScore()
                self.agent.exportQTable(episode)

        env.close()

    def calculateAvgScore(self):
        score_arr = np.array(self.avgScore)
        return np.mean(score_arr)

    def resetAvgScore(self):
        self.avgScore = []


if __name__ == '__main__':
    qLearningAgent = QLearningAgents(alpha=0.01, gamma=0.99, epsilon=0.02)
    # qLearningAgent.importQTable('finalized_model_13000.sav')
    reflexAgents = ReflexAgents()

    # Import Q-Value from previous Training
    game = MountainCart(qLearningAgent, show_every_ep=1000)
    game.runOpenAi()