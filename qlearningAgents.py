import agents
import util


class QLearningAgents(agents):
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
      self.alpha = alpha
      self.gamma = gamma
      self.epsilon = epsilon
      self.values = util.Counter()


   def getQValue(state, action):
      """
      Return Q(s,a)

      :param state:
      :param action:
      :return:
      """

   def computeValueFromQValues(self, state):
      """
      Return max_{a} Q(s,a)

      :param state:
      :return:
      """

   def computeActionFromQValues(self, state):
      """
      Return argmax_{a} Q(s,a) I.e Policy Extraction

      :param state:
      :return:
      """

   def update(self, state, action, newState, reward):
      """
      Update the Q-Table

      :param state:
      :param action:
      :param newState:
      :param reward:
      :return:
      """

   def getAction(selfs, state):
      """

      :param state:
      :return:
      """
      util.raiseNotDefined()
