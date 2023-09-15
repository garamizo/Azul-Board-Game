import numpy as np

from random import choice, shuffle, random, randint, choices
# from numba import njit
# try:
#     from utils import deepcopy
# except ModuleNotFoundError:
#     from azul.utils import deepcopy
# from copy import deepcopy

NUM_LINES = 5
NUM_COLORS = 5
NUM_FLOOR = 7
GRID_PATTERN = [
    [1, 2, 3, 4, 5],
    [5, 1, 2, 3, 4],
    [4, 5, 1, 2, 3],
    [3, 4, 5, 1, 2],
    [2, 3, 4, 5, 1]]
FLOOR_PATTERN = [
    -1, -1, -2, -2, -2, -3, -3]
FLOOR_PATTERN_CUM = [0, -1, -2, -4, -6, -8, -11, -14]

NUM_FACTORY_VS_PLAYER = {2: 5, 3: 7, 4: 9}
NUM_COLORS_PER_FACTORY = 4
NUM_TOTAL_TILES = 100
PTS_PER_ROW = 2
PTS_PER_COL = 7
PTS_PER_SET = 10


class Board:

    def __init__(self):
        self.score = 0
        self.grid = np.zeros([NUM_COLORS, NUM_COLORS], dtype=int)
        self.line = [[] for i in range(NUM_COLORS)]
        self.floor = []

    def step_round(self):
        # update score and return discarded tiles

        def score_move(grid, row, col):

            def count(rowDir, colDir):
                for i in range(1, NUM_COLORS + 1):
                    r, c = row + i * rowDir, col + i * colDir
                    if (
                        r < 0
                        or c < 0
                        or r >= NUM_COLORS
                        or c >= NUM_COLORS
                        or grid[r, c] == 0
                    ):
                        return i-1

            hpts = count(1, 0) + count(-1, 0) + 1
            vpts = count(0, 1) + count(0, -1) + 1
            pts = hpts + vpts
            if hpts == 1 or vpts == 1:
                pts -= 1

            return pts

        discard = [] + self.floor

        for row, line in enumerate(self.line):
            if len(line) == row+1:
                mark = line[0]
                col = GRID_PATTERN[row].index(mark)
                self.grid[row, col] = mark
                self.score += score_move(self.grid, row, col)
                discard += line[1:]
                self.line[row] = []

        # disregard 8th+ floor tiles
        self.score += sum(FLOOR_PATTERN[:min(7, len(self.floor))])
        if self.score < 0:
            self.score = 0
        self.floor = []

        return discard

    def print(self):
        print(self.line)
        print(self.grid)
        print(self.floor)
        print(self.score)
        print('---')

    def is_valid(self, mark, row):
        if row == -1:
            return True
        return (
            len(self.line[row]) <= row
            and (len(self.line[row]) == 0 or self.line[row][0] == mark)
            and np.all(self.grid[row] != mark)
        )

    def step_move(self, mark, numMark, row, isFirst=False):

        assert self.is_valid(mark, row), "Invalid move"

        if isFirst:
            self.floor.append(-1)

        if row == -1:
            self.floor += [mark] * numMark
            return

        dropPieces = numMark + len(self.line[row]) - row - 1
        if dropPieces > 0:
            self.line[row] = [mark] * (row + 1)
            self.floor += [mark] * dropPieces
        else:
            self.line[row] += [mark] * numMark

    def score_board(self):
        finalScore = self.score
        for i in range(NUM_COLORS):
            if np.all(self.grid[i, :] != 0):
                finalScore += PTS_PER_ROW
            if np.all(self.grid[:, i] != 0):
                finalScore += PTS_PER_COL
            if np.sum(self.grid == i+1) == NUM_COLORS:
                finalScore += PTS_PER_SET
        return finalScore


class Table:
    activePlayer = 0

    def __init__(self, numPlayers=3):
        self.numPlayers = numPlayers
        self.bag = [NUM_TOTAL_TILES // NUM_COLORS] * NUM_COLORS
        self.discard = [0] * NUM_COLORS

        self.player = [Board() for _ in range(self.numPlayers)]
        self.numFactories = NUM_FACTORY_VS_PLAYER[self.numPlayers]
        self.roundIdx = 0
        self.reset()

        # full board ===========================
        # p = 0
        # self.player[p].grid = np.array([
        #     [1, 2, 3, 4, 5],
        #     [0, 1, 0, 3, 0],
        #     [4, 0, 1, 0, 3],
        #     [0, 4, 0, 1, 2],
        #     [0, 3, 4, 0, 0]])
        # self.player[p].line = [
        #     [], [2], [5, 5], [], [1, 1, 1, 1]
        # ]
        # self.player[p].floor = []
        # self.player[p].score = 0

        # p = 1
        # self.player[p].grid = np.array([
        #     [1, 2, 3, 4, 5],
        #     [0, 1, 0, 3, 0],
        #     [4, 0, 1, 0, 3],
        #     [0, 4, 0, 1, 2],
        #     [0, 3, 4, 0, 0]])
        # self.player[p].line = [
        #     [], [2], [5, 5], [], [1, 1, 1, 1]
        # ]
        # self.player[p].floor = []
        # self.player[p].score = 0

        # self.factory = [[]] * self.numFactories
        # self.center = [1] + [2] + [3] + [4] + [5]*1
        # self.discard = [1]

        # sim end of round ======================
        # self.factory = [[]] * self.numFactories
        # self.center = [-1] + [1]*2 + [2]*3 + [3]*2
        # self.center = [1, 2]

        # sim end of game =======================
        # p = 0
        # self.factory = [[]] * self.numFactories
        # self.player[p].grid = np.array([
        #     [1, 2, 3, 4, 5],
        #     [0, 1, 0, 3, 4],
        #     [4, 0, 1, 0, 3],
        #     [0, 4, 0, 1, 2],
        #     [0, 3, 4, 0, 0]])
        # self.player[p].line = [
        #     [], [2, 2], [5, 5], [], [1, 1, 1, 1, 1]
        # ]
        # self.player[p].floor = [-1, 2, 4]
        # self.player[p].score = 10
        # self.center = [1]*2 + [2]*2 + [3]*2
        # state = {'factory': [[1, 2, 3, 4], [2, 2, 2, 1], [5, 3, 5, 3], [4, 3, 1, 5], [4, 2, 2, 3], [1, 4, 4, 1], [5, 4, 2, 5]], 'center': [-1], 'player': [{'grid': [[1, 2, 3, 0, 0], [5, 1, 2, 0, 0], [4, 5, 1, 0, 0], [3, 4, 0, 0, 0], [2, 0, 0, 0, 0]], 'line': [[], [], [], [], [4, 4, 4, 4]], 'score': 35, 'floor': []}, {'grid': [[0, 2, 3, 4, 5], [5, 1, 0, 3, 4], [0, 5, 0,
        # round 4
        # state = {'factory': [[1, 1, 2, 3], [1, 5, 2, 2], [4, 1, 4, 2], [3, 2, 4, 2], [4, 1, 3, 4], [5, 5, 1, 3], [3, 1, 3, 4]], 'center': [-1], 'player': [{'grid': [[1, 2, 3, 4, 0], [0, 1, 2, 3, 4], [0, 5, 1, 2, 0], [0, 4, 5, 1, 0], [0, 0, 0, 5, 1]], 'line': [[], [], [3, 3], [], []], 'score': 51, 'floor': []}, {'grid': [[0, 2, 3, 4, 5], [5, 0, 2, 3, 4], [
        #     0, 0, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [1, 1], [3, 3]], 'score': 34, 'floor': []}, {'grid': [[0, 2, 3, 4, 5], [0, 1, 2, 3, 4], [0, 5, 1, 2, 3], [0, 4, 5, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], [2]], 'score': 48, 'floor': []}], 'activePlayer': 0, 'roundIdx': 4, 'bag': [0, 1, 1, 1, 1], 'discard': [3, 3, 1, 5, 8]}
        # round 3
        # state = {'factory': [[3, 3, 4, 2], [5, 5, 2, 3], [5, 1, 2, 5], [5, 2, 3, 2], [3, 5, 3, 5], [2, 2, 1, 1], [4, 5, 3, 1]], 'center': [-1], 'player': [{'grid': [[0, 0, 3, 4, 0], [0, 0, 2, 3, 0], [0, 0, 1, 2, 0], [0, 0, 5, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], [4, 4, 4]], 'score': 13, 'floor': []}, {'grid': [[0, 0, 0, 4, 5], [0, 1, 0, 0, 4], [4, 0, 0, 0, 0], [
        #     0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], [3, 3, 3, 3]], 'score': 8, 'floor': []}, {'grid': [[0, 0, 0, 4, 5], [0, 0, 0, 3, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [5, 5], [2, 2, 2], [1, 1]], 'score': 11, 'floor': []}], 'activePlayer': 0, 'roundIdx': 2, 'bag': [2, 2, 3, 5, 4], 'discard': [9, 5, 3, 4, 3]}
        # state = {'factory': [[2, 4, 4, 1], [3, 3, 5, 3], [3, 1, 4, 1], [1, 2, 2, 4], [5, 1, 5, 2], [3, 1, 5, 4], [2, 5, 4, 4]], 'center': [-1], 'player': [{'grid': [[0, 2, 3, 4, 0], [5, 1, 2, 0, 0], [4, 5, 1, 0, 0], [0, 4, 0, 0, 0], [0, 3, 0, 0, 1]], 'line': [[], [], [], [3, 3], []], 'score': 24, 'floor': []}, {'grid': [[1, 2, 3, 0, 5], [5, 1, 2, 0, 4], [
        #     0, 0, 0, 2, 3], [0, 0, 0, 1, 2], [0, 0, 4, 5, 0]], 'line': [[], [], [], [], []], 'score': 31, 'floor': []}, {'grid': [[1, 2, 3, 0, 5], [5, 0, 2, 3, 4], [0, 5, 1, 0, 0], [0, 4, 0, 0, 2], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], [3, 3]], 'score': 22, 'floor': []}], 'activePlayer': 0, 'roundIdx': 4, 'bag': [2, 1, 5, 1, 2], 'discard': [4, 5, 0, 5, 5]}
        # state = {'factory': [[2, 4, 4, 1], [], [], [], [5, 1, 5, 2], [3, 1, 5, 4], [2, 5, 4, 4]], 'center': [-1, 1, 4, 5, 3, 4], 'player': [{'grid': [[0, 2, 3, 4, 0], [5, 1, 2, 0, 0], [4, 5, 1, 0, 0], [0, 4, 0, 0, 0], [0, 3, 0, 0, 1]], 'line': [[], [], [], [3, 3], [2, 2]], 'score': 24, 'floor': []}, {'grid': [[1, 2, 3, 0, 5], [5, 1, 2, 0, 4], [0, 0, 0, 2, 3], [
        #     0, 0, 0, 1, 2], [0, 0, 4, 5, 0]], 'line': [[], [3, 3], [], [], []], 'score': 31, 'floor': [3]}, {'grid': [[1, 2, 3, 0, 5], [5, 0, 2, 3, 4], [0, 5, 1, 0, 0], [0, 4, 0, 0, 2], [0, 0, 0, 0, 0]], 'line': [[], [1, 1], [], [], [3, 3]], 'score': 22, 'floor': []}], 'activePlayer': 0, 'roundIdx': 4, 'bag': [2, 1, 5, 1, 2], 'discard': [4, 5, 0, 5, 5]}

        # state = {'factory': [[4, 5, 3, 1], [2, 4, 3, 2], [3, 2, 2, 3], [5, 2, 4, 3], [2, 4, 1, 2], [3, 2, 2, 3], [4, 4, 1, 4]], 'center': [-1], 'player': [{'grid': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], []], 'score': 0, 'floor': []}, {'grid': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [
        #     0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], []], 'score': 0, 'floor': []}, {'grid': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'line': [[], [], [], [], []], 'score': 0, 'floor': []}], 'activePlayer': 0, 'roundIdx': 0, 'bag': [17, 11, 13, 13, 18], 'discard': [0, 0, 0, 0, 0]}
        # self.load_state(state)

    def draw_tiles(self, nTiles):
        tiles = []
        for _ in range(nTiles):
            idx = randint(0, sum(self.bag) - 1)
            for color, numTiles in enumerate(self.bag):
                idx -= numTiles
                if idx < 0:
                    tiles.append(color+1)
                    self.bag[color] -= 1
                    break
        return tiles

    def load_state(self, state):
        self.factory = state['factory']
        self.center = state['center']
        self.player = [Board() for _ in range(self.numPlayers)]
        for i, p in enumerate(state['player']):
            self.player[i].grid = np.array(p['grid'])
            self.player[i].line = p['line']
            self.player[i].floor = p['floor']
            self.player[i].score = p['score']
        self.activePlayer = state['activePlayer']
        self.roundIdx = state['roundIdx']
        self.bag = state['bag']
        self.discard = state['discard']

    def reset(self):
        # reshuffle tiles from discard
        # check if sum of values in bag is enough
        if sum(self.bag) < self.numFactories * NUM_COLORS_PER_FACTORY:
            for i in range(NUM_COLORS):
                self.bag[i] += self.discard[i]
                self.discard[i] = 0

            assert sum(self.bag) >= self.numFactories * \
                NUM_COLORS_PER_FACTORY, "missing tile [2]"

        # deal tiles on factories
        self.factory = [[]] * self.numFactories
        for i in range(self.numFactories):
            self.factory[i] = self.draw_tiles(NUM_COLORS_PER_FACTORY)
        self.center = [-1]

    def print(self):
        for factory in self.factory:
            print(factory)
        print(" ", self.center)
        for player in self.player:
            player.print()
        print("\n")

    def step_move(self, mark, factoryIdx, row):
        # step game for a single player move

        factory = self.factory[factoryIdx] if factoryIdx >= 0 else self.center
        player = self.player[self.activePlayer]
        numMark = factory.count(mark)

        assert numMark > 0, "Mark not available"
        assert player.is_valid(mark, row), "Board cannot fit mark"

        isFirst = (factory.count(-1) > 0) and (factoryIdx == -1)
        player.step_move(mark, numMark, row, isFirst)

        for _ in range(numMark):
            factory.remove(mark)
        if isFirst:
            factory.remove(-1)
        elif factoryIdx >= 0:
            self.center += factory
            factory.clear()

        self.activePlayer = (self.activePlayer + 1) % self.numPlayers

        if self.is_round_over():
            print("--- Round over")
            self.roundIdx += 1

            # update score, get discards, find next first player
            for i, player in enumerate(self.player):
                discardi = player.step_round()
                if discardi.count(-1) > 0:
                    self.activePlayer = i
                    discardi.remove(-1)
                for color in discardi:
                    self.discard[color-1] += 1

            if self.is_game_over():
                print("--- Game over")
                for i, player in enumerate(self.player):
                    print(f"Player {i} scores {player.score_board()}")
                    player.score = player.score_board()
            else:
                self.reset()

    def get_observation(self):
        # activePlayer, isGameOver
        # rollout(obs, move) -> [] (return final score of each player)
        #   is_valid(obs, move) -> bool
        #   valid_moves(obs) -> []
        #   update_valid_moves(obs, [moves], move) -> []
        #   is_game_over(obs) -> bool
        obs = {
            'factory': self.factory,
            'center': self.center,
            'player': [],
            'activePlayer': self.activePlayer,
            'roundIdx': self.roundIdx,
            'bag': self.bag,
            'discard': self.discard
        }
        obs['player'] = [
            {'grid': self.player[i].grid.tolist(),
             'line': self.player[i].line,
             'score': self.player[i].score,
             'floor': self.player[i].floor} for i in range(self.numPlayers)]
        return deepcopy(obs)

    def step_round(self, playerPolicyFcn, startPlayerIdx=0):
        # loop through player moves until factories and center are empty
        #  playerPolicyFcn: (playerIdx, obs) -> mark, factoryIdx, row

        gameOver = self.is_game_over()
        playerIdx = startPlayerIdx
        while not gameOver:

            while not self.is_round_over():

                obs = self.get_observation()
                mark, factoryIdx, row = playerPolicyFcn[playerIdx](
                    playerIdx, obs)
                self.step_move(mark, factoryIdx, playerIdx, row)
                # change to next player
                # playerIdx = (playerIdx + 1) % self.numPlayers
                playerIdx = self.activePlayer

            # update score, get discards, find next first player
            for i, player in enumerate(self.player):
                discardi = player.step_round()
                if discardi.count(-1) > 0:
                    playerIdx = i
                    discardi.remove(-1)
                for color in discardi:
                    self.discard[color-1] += 1

            gameOver = self.is_game_over()
            if not gameOver:
                self.reset()

    def is_valid(self, mark, factoryIdx, row):
        player = self.player[self.activePlayer]
        factory = self.factory[factoryIdx] if factoryIdx >= 0 else self.center
        numMark = factory.count(mark)

        # valid for factory
        if numMark == 0:
            return False

        # valid for player
        if row == -1:
            return True
        return (
            len(player.line[row]) <= row
            and (len(player.line[row]) == 0 or player.line[row][0] == mark)
            and np.all(player.grid[row] != mark)
        )

    def is_round_over(self):
        # true if all factories and center are empty
        if len(self.center) > 0:
            return False
        for factory in self.factory:
            if len(factory) > 0:
                return False
        return True

    def is_game_over(self):
        # call after is_round_over, true if a player has a completed row
        if self.is_round_over() == False:
            return False
        for player in self.player:
            if np.any(np.all(player.grid != 0, 1)):
                return True
        return False


def is_game_over(state):
    """
        If game reached end of round ie. factories and center are empty
        Does not consider other rounds
    """
    if len(state['center']) > 0:
        return False
    for factory in state['factory']:
        if len(factory) > 0:
            return False
    for player in state['player']:
        for g in player['grid']:
            if g.count(0) == 0:
                return True
    return False


def score_move(grid, row, col):

    def count(rowDir, colDir):
        for i in range(1, NUM_COLORS + 1):
            r, c = row + i * rowDir, col + i * colDir
            if (
                r < 0
                or c < 0
                or r >= NUM_COLORS
                or c >= NUM_COLORS
                or grid[r][c] == 0
            ):
                return i-1

    hpts = count(1, 0) + count(-1, 0) + 1
    vpts = count(0, 1) + count(0, -1) + 1
    pts = hpts + vpts
    if hpts == 1 or vpts == 1:
        pts -= 1

    return pts


def score_grid(grid):
    """
        Calc additional score from grid patterns
    """
    finalScore = 0
    zVertCount = [0] * NUM_COLORS
    cCount = [0] * NUM_COLORS
    for gRow in grid:
        if gRow.count(0) == 0:
            finalScore += PTS_PER_ROW

        for i in range(NUM_COLORS):
            if gRow[i] != 0:
                zVertCount[i] += 1
            if (i+1) in gRow:
                cCount[i] += 1

    for i in range(NUM_COLORS):
        if zVertCount[i] >= NUM_COLORS:
            finalScore += PTS_PER_COL
        if cCount[i] >= NUM_COLORS:
            finalScore += PTS_PER_SET

    return finalScore


def is_terminal(state):
    """
        If game reached end of round ie. factories and center are empty
        Does not consider other rounds
        Use internally
    """
    if len(state['center']) > 0:
        return False
    for factory in state['factory']:
        if len(factory) > 0:
            return False
    return True


def get_random_action(state):
    # actions, _ = get_action_space(state)
    # return choice(actions)
    player = state['player'][state['activePlayer']]
    while True:
        # factoryIdx = randint(-1, len(state['factory']) - 1)
        factoryIdx = int(random() * (len(state['factory']) + 1)) - 1
        if factoryIdx >= 0:
            factory = state['factory'][factoryIdx]
        else:
            factory = state['center']

        if len(factory) == 0:
            continue

        # color = randint(1, NUM_COLORS)
        color = int(random() * NUM_COLORS) + 1
        if color not in factory:
            continue

        # row = randint(-1, NUM_LINES - 1)
        row = int(random() * (NUM_LINES + 1)) - 1
        if row == -1 or (
            len(player['line'][row]) <= row
            and (len(player['line'][row]) == 0 or player['line'][row][0] == color)
            and (color not in player['grid'][row])
        ):
            return (factoryIdx, color, row)


def is_valid_action(state, action):
    """
        Test if action is valid
        action: (factoryIdx, color, row)
    """
    factoryIdx, color, row = action
    playerIdx = state['activePlayer']
    player = state['player'][playerIdx]
    factory = state['factory'][factoryIdx] if factoryIdx >= 0 else state['center']
    numMark = factory.count(color)

    # valid for factory
    if numMark == 0:
        return False

    # valid for player
    if row == -1:
        return True
    return (
        len(player['line'][row]) <= row
        and (len(player['line'][row]) == 0 or player['line'][row][0] == color)
        and np.all(player['grid'][row] != color)
    )


def get_reward_heuristic(obs):
    """ given state, return [reward]*numPlayer for MC approach
        Assume end of round ie. factories and center are empty, completed lines were cleared, score added
        Modify obs
        Stochastic
    """
    # tiles remaining to complete a row
    # obs = reset_round(obs)

    DISCOUNT = 0.5
    POINT_REWARD = 0.01
    roundsLeft = min([min([g.count(0) for g in p['grid']])
                     for p in obs['player']])
    # print(f"{roundsLeft=}")

    if roundsLeft > 0:
        obs['player'][obs['activePlayer']]['score'] += 4 / roundsLeft
        for p in obs['player']:
            # heuristics on state of grid and lines ==================
            vertCount = [0] * NUM_COLORS  # tile count per column
            cCount = [0] * NUM_COLORS  # tile count per color
            tCount = 0  # total tile count
            # hCount = []  # just for debug
            for gRow in p['grid']:
                horzCount = NUM_COLORS - gRow.count(0)
                # hCount.append(horzCount)
                if horzCount + roundsLeft >= NUM_COLORS:
                    p['score'] += PTS_PER_ROW * \
                        DISCOUNT * horzCount / NUM_COLORS

                for i in range(NUM_COLORS):
                    if gRow[i] != 0:
                        vertCount[i] += 1
                        tCount += 1
                    if i+1 in gRow:
                        cCount[i] += 1

            p['score'] += tCount * 1 * DISCOUNT
            for i in range(NUM_COLORS):
                p['score'] += PTS_PER_COL * DISCOUNT * \
                    (vertCount[i]**2 / NUM_COLORS) * \
                    (roundsLeft / NUM_COLORS)
                p['score'] += PTS_PER_SET * DISCOUNT * \
                    (cCount[i]**1.5 / NUM_COLORS) * (roundsLeft / NUM_COLORS)

            # print(hCount, PTS_PER_ROW * DISCOUNT / NUM_COLORS, '-',
            #       vertCount, PTS_PER_COL * DISCOUNT * roundsLeft / NUM_COLORS**2, '-',
            #       cCount, PTS_PER_SET * DISCOUNT * roundsLeft / NUM_COLORS**2, '-',
            #       tCount, 1 * DISCOUNT)

    scores = [p['score'] for p in obs['player']]
    # allow too high scores, including letting me win
    # return [1 + (r - max(scores)) / (1 + max(scores)) for r in scores]
    # return [1 + (r - max(scores)) / 100 for r in scores]
    # player losing badly loses motivation and plays randomly
    # return [1 if s == max(scores) else 0 for s in scores]
    # still low motivation, but better
    maxScore = max(scores)
    numWinners = scores.count(maxScore)
    sumScore = sum(scores)
    # return [1/numWinners if s == maxScore else 0.1*s/(maxScore + 1) for s in scores]
    return [(s == maxScore)/numWinners + POINT_REWARD*s/sumScore for s in scores]
    # return [(s == max(scores)) + s/1_000 for s in scores]
    # return [1 if s == max(scores) else 0.1*s/(max(scores) + 1) for s in scores]
    # return [r / (sum(scores) + 1) for r in scores]
    # return scores


def get_reward(obs):
    """ given state, return [reward]*numPlayer for MC approach
        Assume end of round ie. factories and center are empty, completed lines were cleared, score added
        Modify obs
        Stochastic
    """
    POINT_REWARD = 0.001
    # if obs['roundIdx'] < 0:
    #     return get_reward_heuristic(obs)

    NUM_REPS = 10

    if is_game_over(obs):
        scores = [p['score'] for p in obs['player']]
        # return [1 if s == max(score) else 0 for s in score]
        maxScore = max(scores)
        numWinners = scores.count(maxScore)
        sumScore = sum(scores) + 1
        # return [(s == maxScore)/numWinners + POINT_REWARD*s/sumScore for s in scores]
        return [(s == maxScore)/numWinners for s in scores]
    # simulate several game outcomes expected value of scores
    else:
        # print(f"{obs['roundIdx']}")
        scores = [0] * len(obs['player'])
        for _ in range(NUM_REPS):
            state = deepcopy(obs)
            state = reset_round(state)
            # roundsLeft = 1

            while not is_game_over(state):  # one more round
                while not is_terminal(state):  # end of round
                    action = get_random_action(state)
                    play(state, action)

                state = reset_round(state)

            score = [p['score'] for p in state['player']]
            maxScore = max(score)
            numWinners = score.count(maxScore)
            sumScore = sum(score) + 1
            for i, s in enumerate(score):
                # scores[i] += ((s == maxScore)/numWinners +
                #               POINT_REWARD*s/sumScore) / NUM_REPS
                scores[i] += ((s == maxScore)/numWinners) / NUM_REPS

        return scores


def get_action_space(state, progressiveBias=False):
    """
        Return:
            ~List of dict [{'factoryIdx': 0, 'colorIdx': 0, 'row': 0}]~
            List of list
    """
    p = state['player'][state['activePlayer']]
    actions = []
    bias = []
    floorLen = len(p['floor'])
    subScore0 = FLOOR_PATTERN_CUM[min(7, floorLen)]

    for factoryIdx, factory in enumerate([state['center']] + state['factory']):
        for color in range(1, NUM_COLORS + 1):
            if color in factory:
                numTiles = factory.count(color)

                for r in range(NUM_LINES):
                    lenLine = len(p['line'][r])
                    if (
                        (0 < lenLine <= r and color in p['line'][r]) or
                        (lenLine == 0 and color not in p['grid'][r])
                    ):
                        actions.append([factoryIdx-1, color, r])

                        numDrops = max(0, numTiles + lenLine - r - 1)
                        subScore = FLOOR_PATTERN_CUM[min(
                            7, floorLen + numDrops)] - subScore0
                        bias.append(numTiles - subScore)

                actions.append([factoryIdx-1, color, -1])
                subScore = FLOOR_PATTERN_CUM[min(
                    7, floorLen + numTiles)] - subScore0
                bias.append(subScore)

    return actions, bias


def reset_round(state=None, numPlayers=2):
    # create new game
    if state is None:
        state = {
            'factory': [[]] * NUM_FACTORY_VS_PLAYER[numPlayers],
            'center': [-1],
            'player': [],
            'activePlayer': 0,
            'roundIdx': 0,
            'discard': [0] * NUM_COLORS,
            'bag': [NUM_TOTAL_TILES // NUM_COLORS] * NUM_COLORS
        }

        for i in range(numPlayers):
            state['player'].append({
                'grid': np.zeros([NUM_LINES, NUM_LINES], int).tolist(),
                'line': np.array([[]] * NUM_LINES).tolist(),
                'score': 0,
                'floor': []})
    else:
        numPlayers = len(state['player'])
        state['roundIdx'] += 1

    numFactory = NUM_FACTORY_VS_PLAYER[numPlayers]

    # for pIdx, p in enumerate(state['player']):
    #     # update board from last round
    #     for row, line in enumerate(p['line']):
    #         # completed line
    #         if len(line) == row+1:
    #             color = line[0]
    #             col = GRID_PATTERN[row].index(color)
    #             p['grid'][row][col] = color
    #             p['score'] += score_move(p['grid'], row, col)
    #             p['line'][row] = []
    #             state['discard'][color-1] += row

    #     # sub floor tiles, disregard 8th+ floor tiles, crop score if negative
    #     p['score'] += FLOOR_PATTERN_CUM[min(7, len(p['floor']))]
    #     for v in p['floor']:
    #         if v == -1:
    #             state['activePlayer'] = pIdx
    #         else:
    #             state['discard'][v-1] += 1
    #     p['floor'].clear()
    #     if p['score'] < 0:
    #         p['score'] = 0

    # if is_game_over(state):
    #     for p in state['player']:
    #         p['score'] += score_grid(p['grid'])

    # prepare factories for next round
    if not is_game_over(state):
        # reshuffle tiles from discard
        if sum(state['bag']) < NUM_COLORS_PER_FACTORY * numFactory:
            # add discard into bag
            state['bag'] = [state['bag'][i] + state['discard'][i]
                            for i in range(NUM_COLORS)]
            state['discard'] = [0] * NUM_COLORS

            assert sum(state['bag']) >= NUM_COLORS_PER_FACTORY * \
                numFactory, "Missing tiles"

        # for each factory sample 4 colors from bag, no repetition
        #  efficient implementation of sampling from bag
        for i, factory in enumerate(state['factory']):
            state['factory'][i] = draw_tiles(
                NUM_COLORS_PER_FACTORY, state['bag'])

    return state


def draw_tiles(nTiles, bag):
    tiles = []
    for _ in range(nTiles):
        idx = randint(0, sum(bag) - 1)
        # idx = int(random() * sum(bag))
        for color, numTiles in enumerate(bag):
            idx -= numTiles
            if idx < 0:
                tiles.append(color+1)
                bag[color] -= 1
                break
    return tiles
    # tiles = []
    # for _ in range(nTiles):
    #     color = choices(range(NUM_COLORS), bag, k=1)
    #     bag[color[0]] -= 1
    #     tiles.append(color[0]+1)
    # return tiles


def play(state, action):
    """"
        Update game state given action
        Does not update score
        No returns
        ~ends at round, not end of game~
        until end of game
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
    else:
        # transfer to lines, and possibly to floor
        dropPieces = numMark + len(player['line'][row]) - row - 1
        if dropPieces > 0:
            player['line'][row] = [mark] * (row + 1)
            player['floor'] += [mark] * dropPieces
        else:
            player['line'][row] += [mark] * numMark

    if is_terminal(state):
        for pIdx, p in enumerate(state['player']):
            # update board from last round
            for row, line in enumerate(p['line']):
                # completed line
                if len(line) == row+1:
                    color = line[0]
                    col = GRID_PATTERN[row].index(color)
                    p['grid'][row][col] = color
                    p['score'] += score_move(p['grid'], row, col)
                    p['line'][row] = []
                    state['discard'][color-1] += row

            # sub floor tiles, disregard 8th+ floor tiles, crop score if negative
            p['score'] += FLOOR_PATTERN_CUM[min(7, len(p['floor']))]
            for v in p['floor']:
                if v == -1:
                    state['activePlayer'] = pIdx
                else:
                    state['discard'][v-1] += 1
            p['floor'].clear()
            if p['score'] < 0:
                p['score'] = 0

        if is_game_over(state):
            for p in state['player']:
                p['score'] += score_grid(p['grid'])


def print_state(state):
    print(f"Bag: {state['bag']}")
    print(f"Discard: {state['discard']}")
    print('Factory:')
    for f in state['factory']:
        print(f)
    print(' ', state['center'])
    for i, p in enumerate(state['player']):
        print(f'Player {i}: {p["score"]}')
        print(' line: ', p['line'])
        print(' grid: ', p['grid'])
        print(' floor: ', p['floor'])


def deepcopy(state):
    obj = state.copy()
    obj['factory'] = state['factory'].copy()
    for i, f in enumerate(state['factory']):
        obj['factory'][i] = f.copy()
    obj['center'] = state['center'].copy()
    obj['bag'] = state['bag'].copy()
    obj['discard'] = state['discard'].copy()

    obj['player'] = state['player'].copy()
    for i, p in enumerate(state['player']):
        obj['player'][i] = p.copy()

        obj['player'][i]['floor'] = p['floor'].copy()
        obj['player'][i]['grid'] = p['grid'].copy()
        obj['player'][i]['line'] = p['line'].copy()
        for r in range(NUM_COLORS):
            obj['player'][i]['grid'][r] = p['grid'][r].copy()
            obj['player'][i]['line'][r] = p['line'][r].copy()

    return obj
