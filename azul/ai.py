import numpy as np
from time import time
from math import log, sqrt
from random import choice
# from numba import njit
try:
    # from utils import deepcopy
    from logic import is_terminal, get_action_space, play, get_reward, deepcopy, get_random_action
except ModuleNotFoundError:
    from azul.logic import is_terminal, get_action_space, play, get_reward, deepcopy, get_random_action
    # from azul.utils import deepcopy

# from azul.tictactoe import is_terminal, get_reward_simple, get_action_space, play, get_reward, REWARD_SUM_MAX


def get_winner(reward):
    maxPts = max(reward)
    if reward.count(maxPts) == 1:
        return reward.index(maxPts)
    else:
        return choice([i for i, val in enumerate(reward) if val == maxPts])


class MCTS_node:
    """
        root: parent is None
        terminators: child is None
    """
    c = sqrt(2)

    def __init__(self, state, parent=None):
        """
            parent: tree parent
            action: player action ie. "move" from self to child
            state: state dict, with at least 'activePlayer'
            board: game state
        """
        self.parent = parent
        self.state = state

        self.numRolls = 0  # num of sims spamming from current node
        self.numWins = 0  # num of wins for parent

        if is_terminal(state):
            self.child, self.action = None, None
        else:
            self.action = get_action_space(state)
            self.child = [None] * len(self.action)

    def select_leaf(self):
        """
            Calculate best branch per UC1 and a 
        """
        node = self
        while None not in node.child:  # node has an unexplored child
            maxVal = -1
            k = MCTS_node.c * sqrt(log(node.numRolls))
            for child in node.child:
                val = child.numWins / child.numRolls + k / sqrt(child.numRolls)
                if val > maxVal:
                    maxVal = val
                    node = child

            if node.child is None:  # terminal node
                return node, None

        # pick one unexplored child randomly
        actionIdx = choice([i for i, n in enumerate(node.child) if n is None])
        return node, actionIdx

    def expand(self, actionIdx):
        stateNew = deepcopy(self.state)
        play(stateNew, self.action[actionIdx])
        self.child[actionIdx] = MCTS_node(stateNew, self)

    def rollout(self):
        """
            Play game randonly until the end
            Assume
            Return:
                reward: [] * numPlayer (that feeds to backpropagate)
        """
        state = deepcopy(self.state)
        while not is_terminal(state):
            action = get_random_action(state)
            play(state, action)

        return get_reward(state)  # call will overwrite 'state'

    def backpropagate(self, rewards):
        # transverse tree to the root, updating win stats
        node = self
        while node.parent is not None:  # until the root
            node.numRolls += 1
            node.numWins += rewards[node.parent.state['activePlayer']]
            node = node.parent
        node.numRolls += 1  # update numSims of root

    def eval(self):
        return self.numWins / self.numRolls + \
            MCTS_node.c * sqrt(log(self.parent.numRolls) / self.numRolls)

    def grow(self):
        node, leafIdx = self.select_leaf()
        if node.child is not None:  # not terminal
            node.expand(leafIdx)
            node = node.child[leafIdx]
        reward = node.rollout()
        node.backpropagate(reward)

    def grow_while(self, timeout=2.0, maxRolls=1_000_000, startTime=None):
        if startTime is None:
            startTime = time()

        # grow tree in alloted time
        while time() < startTime + timeout and self.numRolls <= maxRolls:
            self.grow()
        while time() < startTime + timeout:
            pass

    @staticmethod
    def search_node(root, state, maxDepth=3):
        # search with limited depth
        if root.state == state:
            return root
        # terminal node or too deep
        if maxDepth <= 0 or root.child is None:
            return None

        for node in root.child:
            if node is not None:
                nodeOut = MCTS_node.search_node(node, state, maxDepth-1)
                if nodeOut is not None:
                    return nodeOut
        return None

    def get_best_action(self):
        winRatios = [node.numWins / node.numRolls for node in self.child]
        actionIdx = winRatios.index(max(winRatios))
        return self.action[actionIdx], actionIdx

    def print(self):
        import numpy as np
        print(
            f"Rolls: {self.numRolls}, Win ratio: {self.numWins/self.numRolls:.2}, Active player: {self.state['activePlayer']}")
        # print(np.reshape(self.board, [6, 7]))
        print("\tAction\t\tUCB\tWRatio\tNumSims")
        if self.child is None:
            print(f"Winner: {get_winner(get_reward(deepcopy(self.state)))}")
        else:
            for a, n in zip(self.action, self.child):
                if n is not None:
                    print(
                        f"\t{a}\t{n.eval():.2f}\t{n.numWins/n.numRolls:.3f}\t{n.numRolls}")
