import sys
import gym
from environment import Environment
import numpy as np
from agents import Agents
from qlearningAgents import QLearningAgents
from reflexAgents import ReflexAgents
import numpy as np

class Breakout:

    def __init__(self, agent, numTraining=100000, show_every_ep=1000):
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


            if episode % self.showEveryEp == 0:
                env = gym.make('BreakoutNoFrameskip-v4', obs_type='ram', render_mode='human')
                self.agent.loadEnvironment(env)
            else:
                env = gym.make('BreakoutNoFrameskip-v4', obs_type='ram', render_mode=None)
                self.agent.loadEnvironment(env)

            obv = env.reset()
            done = False
            preprocess_env = Environment(obv)

            dis_obv = preprocess_env.get_discrete_state_breakout_v1(obv, obv)

            # print(f'The initial observation of the block {obv[0:72]}')

            while not done:
                # Retrieve action from agent
                action = self.agent.getAction(dis_obv)

                prev_obv = obv

                # Get new Observation from the environment
                new_obv, reward, done, info = env.step(action)
                dis_new_obv = preprocess_env.get_discrete_state_breakout_v1(prev_obv, new_obv)

                # Update Agent
                self.agent.update(dis_obv, action, dis_new_obv, reward)
                dis_obv = dis_new_obv
                obv = new_obv

                # print(f'The paddle x position is {obv[72]} and the ball x,y position ({obv[99]},{obv[101]}) taking action {action}')

                if reward >= 1:
                    old_state = np.array(prev_obv[0:72])
                    new_state = np.array(obv[0:72])
                    # print(f'HITTTTTT The paddle x position is {obv[72]} and the ball x,y position ({obv[99]},{obv[101]})')
                    # print(f'The old state is \n {old_state} \n and the new state is \n {new_state} \n and The difference is \n {new_state - old_state}')

            self.avgScore.append(new_obv[84])

            if idx % self.showEveryEp == 0:
                avg_score = self.calculateAvgScore()
                print(f'The average score {avg_score} for interval {episode - self.showEveryEp} to {episode}')
                self.resetAvgScore()
                # self.agent.exportQTable(episode)

        env.close()

    def calculateAvgScore(self):
        score_arr = np.array(self.avgScore)
        return np.mean(score_arr)

    def resetAvgScore(self):
        self.avgScore = []


if __name__ == '__main__':
    qLearningAgent = QLearningAgents(alpha=0.05, gamma=0.8, epsilon=0.2)
    reflexAgents = ReflexAgents()

    # Import Q-Value from previous Training
    game = Breakout(reflexAgents)
    game.runOpenAi()