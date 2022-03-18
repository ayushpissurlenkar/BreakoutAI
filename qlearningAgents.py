from agents import Agents
import util
import random


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

   def __init__(self, env, alpha, gamma, epsilon):
      self.env = env
      self.alpha = alpha
      self.gamma = gamma
      self.epsilon = epsilon
      self.values = util.Counter()


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

      print("Opt_Avaliable ", opt_avaliable)
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
      # print("From Q Update Q table ", self.values)


   def getAction(self, state):
      """

      :param state:
      :return:
      """

      if util.flipCoin(self.epsilon):
         action = self.env.action_space.sample()
         print("Random Action ", action)
         return action
      else:
         action = self.computeActionFromQValues(state)
         print("Optimal Action ", action)
         return action


