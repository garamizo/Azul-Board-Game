import numpy as np
from copy import deepcopy
from random import choice, shuffle, random
from copy import deepcopy, copy


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
                # col = np.where(GRID_PATTERN[row] == mark)[0]
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
    numPlayers = 4
    activePlayer = 1

    def __init__(self):
        self.bag = np.random.permutation(
            np.tile(np.arange(1, NUM_COLORS+1), NUM_TOTAL_TILES // NUM_COLORS))
        self.discard = []
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
        #     [0, 4, 0, 0, 2],
        #     [0, 3, 4, 0, 0]])
        # self.player[p].line = [
        #     [0], [2, 2], [5, 5], [], [1, 1, 1, 1, 1]
        # ]
        # self.player[p].floor = [-1, 2, 4]
        # self.player[p].score = 10
        # self.center = [1]*2 + [2]*2 + [3]*2

    def reset(self):
        # reshuffle tiles from discard
        if self.bag.shape[0] < self.numFactories * NUM_COLORS_PER_FACTORY:
            # self.bag = np.random.permutation(self.discard)
            self.bag = np.random.permutation(
                np.tile(np.arange(1, NUM_COLORS+1), NUM_TOTAL_TILES // NUM_COLORS))

        # deal tiles on factories
        self.factory = [[]] * self.numFactories
        for i in range(self.numFactories):
            self.factory[i], self.bag = self.bag[:4].tolist(), self.bag[4:]

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
            self.roundIdx += 1
            isGameOver = self.is_game_over()
            # update score, get discards, find next first player
            for i, player in enumerate(self.player):
                discardi = player.step_round()
                if discardi.count(-1) > 0:
                    self.activePlayer = i
                self.discard += discardi
                if isGameOver:
                    player.score = player.score_board()

            if not self.is_game_over():
                # self.discard.remove(-1)
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
                self.discard += discardi

            gameOver = self.is_game_over()
            if not gameOver:
                self.discard.remove(-1)
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
        for player in self.player:
            if np.any(np.all(player.grid != 0, 1)):
                return True
        return False


# def valid_move(obs, factoryIdx, mark, row):
#     #
#     playerIdx = obs['activePlayer']
#     player = obs['player'][playerIdx]
#     factory = obs['factory'][factoryIdx] if factoryIdx >= 0 else obs['center']
#     numMark = factory.count(mark)

#     # valid for factory
#     if numMark == 0:
#         return False

#     # valid for player
#     if row == -1:
#         return True
#     return (
#         len(player['line'][row]) <= row
#         and (len(player['line'][row]) == 0 or player['line'][row][0] == mark)
#         and np.all(player['grid'][row] != mark)
#     )


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


def score_board(grid):
    grid = np.array(grid)
    finalScore = 0
    for i in range(NUM_COLORS):
        if np.all(grid[i, :] != 0):
            finalScore += PTS_PER_ROW
        if np.all(grid[:, i] != 0):
            finalScore += PTS_PER_COL
        if np.sum(grid == i+1) == NUM_COLORS:
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


def is_game_over(state):
    # call after is_round_over, true if a player has a completed row
    for player in state['player']:
        for gridRow in player['grid']:
            if gridRow.count(0) == 0:
                return True
    return False


def get_reward(obs):
    """ given state, return [reward]*numPlayer for MC approach
        Assume end of round ie. factories and center are empty, completed lines were cleared, score added
        Modify obs
        Stochastic
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

    rewards = [p['score'] for p in obs['player']]
    return [r / (sum(rewards) + 1) for r in rewards]


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


def get_reward_deterministic(obs):
    """ given state, return [reward]*numPlayer for MC approach
        Assume end of round ie. factories and center are empty, completed lines were cleared, score added
        Modify obs
        Deterministic
    """
    # rounds left == tiles remaining to complete a row
    roundsLeft = min([min([g.count(0) for g in p['grid']])
                     for p in obs['player']])

    # weight function for collumn and set scoring
    wCol = [1/15, 2/15, 3/15, 4/15, 5/15]
    # dict weight for missing tiles to complete row
    rCol = {i: (roundsLeft/(roundsLeft + i) if i <= roundsLeft else 0)
            for i in range(NUM_COLORS)}

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


def get_reward_simple(obs):
    """ given state, return [reward]*numPlayer for MC approach
        Assume end of round ie. factories and center are empty, completed lines were cleared, score added
        Modify obs
        Deterministic
    """
    # rounds left == tiles remaining to complete a row
    isTerminal = is_terminal(obs)
    roundsLeft = min([min([g.count(0) for g in p['grid']])
                     for p in obs['player']])
    # print(f"{roundsLeft=}")
    # weight function for collumn and set scoring
    wCol = [1/15, 2/15, 3/15, 4/15, 5/15]

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

        p['score'] += score_board(np.array(p['grid'], int))

        # sub floor tiles, disregard 8th+ floor tiles, crop score if negative
        p['score'] += FLOOR_PATTERN_CUM[min(7, len(p['floor']))]
        if p['score'] < 0:
            p['score'] = 0

        if roundsLeft == 0 and isTerminal == True:
            continue

        # heuristics on state of grid and lines ==================
        for r, gRow in enumerate(p['grid']):
            numTiles = NUM_LINES - gRow.count(0)
            if numTiles < NUM_LINES:
                p['score'] += numTiles / NUM_LINES * \
                    sum(wCol[r+1:]) * (PTS_PER_ROW + NUM_LINES)

        grid = np.array(p['grid'])
        for gCol in grid.T:
            gColFill = gCol > 0
            if not np.all(gColFill):
                p['score'] += (gColFill @ wCol) * (PTS_PER_COL + NUM_LINES)

        for color in range(1, NUM_COLORS + 1):
            numTiles = np.sum(grid == color)
            missRow = np.any(grid == color, 1)
            if numTiles < NUM_LINES:
                p['score'] += numTiles / NUM_COLORS * \
                    (missRow @ wCol) * PTS_PER_SET

    return [p['score'] for p in obs['player']]


def get_action_space(state):
    """
        Return:
            ~List of dict [{'factoryIdx': 0, 'colorIdx': 0, 'row': 0}]~
            List of list
    """
    player = state['player'][state['activePlayer']]
    actions = []

    for factoryIdx, factory in enumerate([state['center']] + state['factory']):
        for color in range(1, NUM_COLORS + 1):
            if color in factory:
                actions.append([factoryIdx-1, color, -1])

                for r, (gridR, lineR) in enumerate(zip(player['grid'], player['line'])):
                    if (0 < len(lineR) <= r and color in lineR) or (len(lineR) == 0 and color not in gridR):
                        actions.append([factoryIdx-1, color, r])
    return actions


def reset_round(state=None, numPlayers=2):
    # create new game
    if state is None:
        reset_round.bag = list(range(1, NUM_COLORS + 1)) * \
            (NUM_TOTAL_TILES // NUM_COLORS)
        shuffle(reset_round.bag)

        state = {
            'factory': [[]] * NUM_FACTORY_VS_PLAYER[numPlayers],
            'center': [-1],
            'player': [],
            'activePlayer': 0,
            'roundIdx': 0,
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

    for p in state['player']:
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
        p['floor'].clear()
        if p['score'] < 0:
            p['score'] = 0

    if is_game_over(state):
        for p in state['player']:
            p['score'] += score_board(p['grid'])

    else:
        # TODO make bag not repeat tiles
        if len(reset_round.bag) < NUM_COLORS_PER_FACTORY * numFactory:
            reset_round.bag = list(range(1, NUM_COLORS + 1)) * \
                (NUM_TOTAL_TILES // NUM_COLORS)
            shuffle(reset_round.bag)

        for i in range(numFactory):
            state['factory'][i] = reset_round.bag[:NUM_COLORS_PER_FACTORY]
            reset_round.bag = reset_round.bag[NUM_COLORS_PER_FACTORY:]

    return state


def play(state, action):
    """"
        Update game state given action
        Does not update score
        No returns
        ends at round, not end of game
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
