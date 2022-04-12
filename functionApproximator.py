import util
from scipy.spatial import distance


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


class TruncateRamApproximator(functionApproximator):

    def getVectorizedFeatures(self, state, action):
        feats = util.Counter()
        x = state[99]
        y = state[101]
        direction_x = state[103]
        direction_y = state[105]

        x_new = x + direction_x
        y_new = y + direction_y

        paddle_position = state[72]

        if action == 2:
            paddle_position = paddle_position + 5
        elif action == 3:
            paddle_position = paddle_position - 5

        blocks = state[0:30]

        for idx, block_num in enumerate(blocks):
            feats[f'mem_{idx}'] = block_num

        feats['paddle_pos'] = paddle_position
        feats['y_pos'] = y_new
        feats['x_pos'] = x_new

        feats['vel_y'] = direction_x
        feats['vel_y'] = direction_y










