import numpy as np
from time import time
from math import log, sqrt
from random import choice
# from numba import njit
from copy import deepcopy
from logic import is_terminal, get_reward_simple, get_action_space, play, get_reward
# from azul.tictactoe import is_terminal, get_reward_simple, get_action_space, play, get_reward, REWARD_SUM_MAX


def get_winner(reward):
    maxPts = max(reward)
    if reward.count(maxPts) == 1:
        return reward.index(maxPts)
    else:
        return choice([i for i, val in enumerate(reward) if val == maxPts])


def maxn(state, depth, alpha):
    currentPlayer = state['activePlayer']
    if is_terminal(state) or depth <= 0:
        return get_reward(state), None

    best = [-np.Inf] * state['numPlayers']
    actionBest = None
    for action in get_action_space(state):
        stateNew = deepcopy(state)
        play(stateNew, action)
        result, _ = maxn(stateNew, depth-1, best[currentPlayer])

        if result[currentPlayer] > best[currentPlayer]:
            best, actionBest = result, action

        if result[currentPlayer] >= REWARD_SUM_MAX - alpha:
            return result, actionBest
    return best, actionBest


def paranoid(state, depth, alpha, beta, rootPlayer):
    activePlayer = state['activePlayer']
    if is_terminal(state) or depth <= 0:
        reward = get_reward(state)[activePlayer]
        return (1 if activePlayer == rootPlayer else -1) * reward, None

    actionBest = None
    for action in get_action_space(state):
        stateNew = deepcopy(state)
        play(stateNew, action)
        if activePlayer == rootPlayer or stateNew['activePlayer'] == rootPlayer:
            reward = -paranoid(stateNew, depth-1, -beta, -alpha, rootPlayer)[0]
        else:
            reward = paranoid(stateNew, depth-1, alpha, beta, rootPlayer)[0]

        if reward > alpha:
            alpha = reward
            actionBest = action

        if alpha >= beta:
            return beta, actionBest

    return alpha, actionBest


def negamax(state, currentPlayer, depth, playerMaxIdx):
    """
        2 player negamax
            depth
            currentPlayer: -1 or 1
    """
    if is_terminal(state) or depth <= 0:
        winnerIdx = get_winner(get_reward_simple(deepcopy(state)))
        return currentPlayer if playerMaxIdx == winnerIdx else -currentPlayer, None
        # return currentPlayer, None

    valMax, actionMax = -np.Inf, None
    for action in get_action_space(state):
        stateNew = deepcopy(state)
        play(stateNew, action)
        val = -negamax(stateNew, -currentPlayer, depth-1, playerMaxIdx)[0]
        if val >= valMax:
            valMax = val
            actionMax = action

    return valMax, actionMax


def alphabeta(state, currentPlayer, depth, alpha, beta, playerMaxIdx):
    """
        2 player negamax
            depth
            currentPlayer: -1 or 1
    """
    if is_terminal(state) or depth <= 0:
        winnerIdx = get_winner(get_reward_simple(deepcopy(state)))
        return currentPlayer if playerMaxIdx == winnerIdx else -currentPlayer, None

    actionMax = None
    for action in get_action_space(state):
        stateNew = deepcopy(state)
        play(stateNew, action)
        val = -alphabeta(stateNew, -currentPlayer, depth -
                         1, -beta, -alpha, playerMaxIdx)[0]
        if val > alpha:
            alpha = val
            actionMax = action

        if alpha >= beta:
            return beta, None

    return alpha, actionMax


def get_random_action(state):
    actions = get_action_space(state)
    return choice(actions)


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
        # TODO mark not necessary. Implied by board and action

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
            # if node.parent.state['activePlayer'] == winner:
            #     node.numWins += 1
            node = node.parent
        node.numRolls += 1  # update numSims of root

    def eval(self):
        return self.numWins / self.numRolls + \
            MCTS_node.c * sqrt(log(self.parent.numRolls) / self.numRolls)

    def grow(self, timeout=2.0, startTime=None):
        if startTime is None:
            startTime = time()

        # grow tree in alloted time
        while time() < startTime + timeout:
            node, leafIdx = self.select_leaf()
            if node.child is not None:  # not terminal
                node.expand(leafIdx)
                node = node.child[leafIdx]
            reward = node.rollout()
            node.backpropagate(reward)

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
            f"{self.numRolls=}, {self.numWins=}, {self.state['activePlayer']=}")
        # print(np.reshape(self.board, [6, 7]))
        print("\tAction\t\tUCB\tWRatio\tNumSims")
        if self.child is None:
            print(f"Winner: {get_winner(get_reward(self.state))}")
        else:
            for a, n in zip(self.action, self.child):
                if n is not None:
                    print(
                        f"\t{a}\t{n.eval():.2f}\t{n.numWins/n.numRolls:.3f}\t{n.numRolls}")


# def agent_uct(obs):
#     global root_

#     startTime = time()
#     TIMEOUT = 5.0

#     # select branch as root per choice of other players
#     root_ = MCTS_node.search_node(root_, obs['state'], maxDepth=4)
#     root_.parent = None

#     # start new tree
#     if root_ is None:
#         root_ = MCTS_node(obs["state"])

#     # grow tree in alloted time
#     root_.grow(TIMEOUT, startTime)

#     # select best branch
#     action, actionIdx = MCTS_node.get_best_action(root_)
#     root_ = root_.child[actionIdx]
#     root_.parent = None

#     return action
