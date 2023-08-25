from time import time
from math import log, sqrt
from random import choice, shuffle, random
# from numba import njit
from numpy import array
from copy import deepcopy
from logic import NUM_MARKS, NUM_LINES, NUM_FLOOR, PTS_PER_COL, PTS_PER_ROW, PTS_PER_SET

import numpy as np

# TODO Assume 2 player game
GRID_PATTERN = [
    [1, 2, 3, 4, 5],
    [5, 1, 2, 3, 4],
    [4, 5, 1, 2, 3],
    [3, 4, 5, 1, 2],
    [2, 3, 4, 5, 1]]
FLOOR_PATTERN_CUM = [0, -1, -2, -4, -6, -8, -11, -14]


def score_move(grid, row, col):

    def count(rowDir, colDir):
        for i in range(1, NUM_MARKS + 1):
            r, c = row + i * rowDir, col + i * colDir
            if (
                r < 0
                or c < 0
                or r >= NUM_MARKS
                or c >= NUM_MARKS
                or grid[r][c] == 0
            ):
                return i-1

    hpts = count(1, 0) + count(-1, 0) + 1
    vpts = count(0, 1) + count(0, -1) + 1
    pts = hpts + vpts
    if hpts == 1 or vpts == 1:
        pts -= 1

    return pts


def score_board(grid):
    finalScore = 0
    for i in range(NUM_MARKS):
        if np.all(grid[i, :] != 0):
            finalScore += PTS_PER_ROW
        if np.all(grid[:, i] != 0):
            finalScore += PTS_PER_COL
        if np.sum(grid == i+1) == NUM_MARKS:
            finalScore += PTS_PER_SET
    return finalScore


def is_terminal(state):
    """
        If game reached end of round ie. factories and center are empty
        Does not consider other rounds
    """
    if len(state['center']) > 0:
        return False
    for factory in state['factory']:
        if len(factory) > 0:
            return False
    return True


def get_reward(obs):
    """ given state, return [reward]*numPlayer for MC approach
        Assume end of round ie. factories and center are empty, completed lines were cleared, score added
        Modify obs
    """
    # probability of getting N tiles of a specific color in a round
    probTile = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.3, 5: 0.2}

    # rounds left == tiles remaining to complete a row
    roundsLeft = min([min([g.count(0) for g in p['grid']])
                     for p in obs['player']])

    for p in obs['player']:

        # update board from last round
        for row, line in enumerate(p['line']):
            # completed line
            if len(line) == row+1:
                color = line[0]
                col = GRID_PATTERN[row].index(color)
                p['grid'][row][col] = color
                p['score'] += score_move(p['grid'], row, col)
                p['line'][row] = []

        # sub floor tiles, disregard 8th+ floor tiles, crop score if negative
        p['score'] += FLOOR_PATTERN_CUM[min(7, len(p['floor']))]
        if p['score'] < 0:
            p['score'] = 0

        # heuristics on state of grid and lines ==================
        for i in range(roundsLeft):
            for row in range(NUM_LINES):
                # use remaining pieces on lines, only 1 color available
                if len(p['line'][row]) > 0:
                    tilesRemaining = row + 1 - len(p['line'][row])
                    color = p['line'][row][0]
                    if probTile[tilesRemaining] > random():
                        col = GRID_PATTERN[row].index(color)
                        p['score'] += score_move(p['grid'], row, col)
                        p['grid'][row][col] = color
                        p['line'][row] = []
                    # force completion on next round
                    else:
                        p['line'][row] = [color] * (row + 1)
                else:
                    colors = [GRID_PATTERN[row][col]
                              for col, g in enumerate(p['grid'][row]) if g == 0]

                    if len(colors) > 0 and (1 - probTile[row + 1])**len(colors) < random():
                        color = choice(colors)
                        col = GRID_PATTERN[row].index(color)
                        p['score'] += score_move(p['grid'], row, col)
                        p['grid'][row][col] = color

        p['score'] += score_board(np.array(p['grid'], dtype=np.int))

    return [p['score'] for p in obs['player']]


def get_winner(reward):
    maxPts = max(reward)
    if reward.count(maxPts) == 1:
        return reward.index(maxPts)
    else:
        return choice([i for i, val in enumerate(reward) if val == maxPts])


def get_action_space(state):
    """
        Return:
            ~List of dict [{'factoryIdx': 0, 'colorIdx': 0, 'row': 0}]~
            List of list
    """
    player = state['player'][state['activePlayer']]
    actions = []

    for factoryIdx, factory in enumerate([state['center']] + state['factory']):
        for color in range(1, NUM_MARKS + 1):
            if color in factory:
                actions.append([factoryIdx-1, color, -1])

                for r, (gridR, lineR) in enumerate(zip(player['grid'], player['line'])):
                    if (0 < len(lineR) <= r and color in lineR) or (len(lineR) == 0 and color not in gridR):
                        actions.append([factoryIdx-1, color, r])
    return actions


def play(state, action):
    """"
        Update game state given action
        Does not update score
        No returns
    """
    factoryIdx, mark, row = action
    factory = state['factory'][factoryIdx] if factoryIdx >= 0 else state['center']
    player = state['player'][state['activePlayer']]
    numMark = factory.count(mark)
    numPlayer = len(state['player'])

    isFirst = factory.count(-1) > 0

    # update factory tiles
    for _ in range(numMark):
        factory.remove(mark)
    if isFirst:
        factory.remove(-1)
    elif factoryIdx >= 0:
        state['center'] += factory
        factory.clear()

    state['activePlayer'] = (state['activePlayer'] + 1) % numPlayer

    # update player board
    if isFirst:
        player['floor'].append(-1)
    if row == -1:
        player['floor'] += [mark] * numMark
        return

    # transfer to lines, and possibly to floor
    dropPieces = numMark + len(player['line'][row]) - row - 1
    if dropPieces > 0:
        player['line'][row] = [mark] * (row + 1)
        player['floor'] += [mark] * dropPieces
    else:
        player['line'][row] += [mark] * numMark


def get_random_action(state):
    actions = get_action_space(state)
    return choice(actions)


class MCTS_node:
    """
        root: parent is None
        terminators: child is None
    """
    c = sqrt(2)

    def __init__(self, state, agentIdx, parent=None):
        """
            parent: tree parent
            action: player action ie. "move" from self to child
            state: state dict, with at least 'activePlayer'
            board: game state
        """
        # TODO mark not necessary. Implied by board and action

        self.parent = parent
        self.state = state    # state after action, list
        self.agentIdx = agentIdx

        self.numRolls = 0  # num of sims spamming from current node
        self.numWins = 0  # num of wins for parent mark (ie num of losses)

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
        action = choice(
            [a for n, a in zip(node.child, node.action) if n is None])
        return node, action

    def expand(self, action):
        stateNew = deepcopy(self.state)
        play(stateNew, action)
        actionIdx = self.action.index(action)
        self.child[actionIdx] = MCTS_node(stateNew, self.agentIdx, self)
        return self.child[actionIdx]

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

    def backpropagate(self, winner):
        # transverse tree to the root, updating win stats
        node = self
        while node.parent is not None:
            node.numRolls += 1
            if node.parent.state['activePlayer'] == winner:
                # if node.agentIdx == winner:
                node.numWins += 1
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
            node, action = self.select_leaf()
            if not node.child == None:  # not terminal
                node = node.expand(action)
            reward = node.rollout()
            node.backpropagate(get_winner(reward))

    @staticmethod
    def search_node(root, state, maxDepth=2):
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


# convert state object to list
def state_object_to_list(obs):
    # activePlayer = obs['activePlayer']
    # center = obs['center']
    # factory = obs['factory']
    # player = []
    # for p in obs['player']:
    #     score = p['score']
    #     grid = p['grid']
    #     line = p['line']
    #     floor = p['floor']
    #     player.append([score, grid, line, floor])

    # return [activePlayer, center, factory, player]
    return obs


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
