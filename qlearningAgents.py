from agents import Agents
import util
import random
import pickle
from reflexAgents import ReflexAgents


class QLearningAgents(Agents):
   """
  Q-Learning Agent

  Function Compute:
     - getQValue: Return the Q Value for each state
     - computeValueFromQValues: Return max_{a} Q(s,a) from legal action
     - computeActionFromQValues: Return action that maximize Q(s,a) argmax_{a} Q(s,a)
     - getAction: Return Action for which agent need to act
     - update: Update the Q_value table
  """

   def __init__(self, alpha, gamma, epsilon):
      self.env = None
      self.alpha = alpha
      self.gamma = gamma
      self.epsilon = epsilon
      self.values = util.Counter()
      self.heatMap = util.Counter()


   def getQValue(self,state, action):
      """
      Return Q(s,a)

      :param state:
      :param action:
      :return: Q-Value given state action paired
      """
      return self.values[(state, action)]


   def computeValueFromQValues(self, state):
      """
      Return max_{a} Q(s,a)

      :param state:
      :return: Return the maximum Q-Value from all possible action given a state
      """
      value = []

      for action in range(self.env.action_space.n):
          value.append(self.getQValue(state, action))

      return max(value)


   def computeActionFromQValues(self, state):
      """
      Return argmax_{a} Q(s,a) I.e Policy Extraction

      :param state:
      :return: Return optimal action that maximize the Q(s,a)
      """
      opt_avaliable = []
      opt_value = self.computeValueFromQValues(state)

      for action in range(self.env.action_space.n):
         if self.getQValue(state, action) == opt_value:
            opt_avaliable.append(action)

      # print("Opt_Avaliable ", opt_avaliable)
      return random.choice(opt_avaliable)

   def update(self, state, action, newState, reward):
      """
      Update the Q-Table

      :param state:
      :param action:
      :param newState:
      :param reward:
      :return: Return a new Q-Value table
      """
      q_value = (1 - self.alpha) * self.getQValue(state, action) + \
                self.alpha * (reward + self.gamma * self.computeValueFromQValues(newState))
      # print("From Q Update ", q_value, reward)
      self.values[(state, action)] = q_value

      # Update HeatMap
      self.heatMap[(state,action)] = self.heatMap[(state,action)] + 1
      # print("From Q Update Q table ", self.values)


   def getAction(self, state):
      """

      :param state:
      :return:
      """

      if util.flipCoin(self.epsilon):
         # Using Reflex Agent to train
         # action = ReflexAgents().getAction(state)
         action = self.env.action_space.sample()
         # print("Random Action ", action)
         return action
      else:
         action = self.computeActionFromQValues(state)
         # print("Optimal Action ", action)
         return action


   def exportQTable(self, episode):
      """ Export the trained model for later reference

      :param episode: represent the episode which is saved
      :return: pickle file of q-value table
      """
      self.values['episode'] = episode
      pickle.dump(self.values, open(f'finalized_model_{episode}.sav', 'wb'))

      pickle.dump(self.heatMap, open(f'heat_map_q_table.sav','wb'))

   def loadEnvironment(self, env):
      """ Given the openAI gym environment load the environment

      :param env: Represent the openAI gym environment
      :return: set the environment for the Q-Learning Agent
      """
      self.env = env

   def importQTable(self, filepath):
      """ Given the model set up the Q-table for the q-learning
          agent

      :param filepath: represent the file path of q-learning model
      :return: set the q-learning agent with loaded model
      """
      self.values = pickle.load(open(filepath, 'rb'))

   def getEpisode(self):
      """ Return the latest episode that is saved

      :return: the epsidoe that is saved
      """
      return self.values['episode']

   def setAlpha(self, alpha):
      self.alpha = alpha






