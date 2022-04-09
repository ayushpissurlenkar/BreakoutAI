import util


class functionApproximator:

    def getVectorizedFeatures(self, state, action):
        """ Given state action pair return the vectorized Feature of the gamestate

        :param state: Represent the state of the game
        :param action: Represent the action taken from that state
        :return: feature
        """

        util.raiseNotDefined()


class IdentityApproximator(functionApproximator):

    def getVectorizedFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats