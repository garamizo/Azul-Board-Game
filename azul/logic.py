import numpy as np

NUM_LINES = 5
NUM_MARKS = 5
NUM_FLOOR = 7
GRID_PATTERN = np.array([
    [1, 2, 3, 4, 5],
    [5, 1, 2, 3, 4],
    [4, 5, 1, 2, 3],
    [3, 4, 5, 1, 2],
    [2, 3, 4, 5, 1]])
FLOOR_PATTERN = np.array([
    -1, -1, -2, -2, -2, -3, -3])
NUM_FACTORY_VS_PLAYER = {2: 5, 3: 7, 4: 9}
NUM_MARKS_PER_FACTORY = 4
NUM_TOTAL_TILES = 100
PTS_PER_ROW = 2
PTS_PER_COL = 7
PTS_PER_SET = 10


class Board:

    def __init__(self):
        self.score = 0
        self.grid = np.zeros([NUM_MARKS, NUM_MARKS])
        self.line = [[] for i in range(NUM_MARKS)]
        self.floor = []

    def step_round(self):
        # update score and return discarded tiles

        def score_move(grid, row, col):

            def count(rowDir, colDir):
                for i in range(1, NUM_MARKS + 1):
                    r, c = row + i * rowDir, col + i * colDir
                    if (
                        r < 0
                        or c < 0
                        or r >= NUM_MARKS
                        or c >= NUM_MARKS
                        or grid[r, c] == 0
                    ):
                        return i-1

            return 2 + count(1, 0) + count(0, 1) + count(-1, 0) + count(0, -1)

        discard = [] + self.floor

        for row, line in enumerate(self.line):
            if len(line) == row+1:
                mark = line[0]
                col = np.where(GRID_PATTERN[row] == mark)[0]
                self.grid[row, col] = mark
                self.score += score_move(self.grid, row, col)
                discard += line[1:]
                self.line[row] = []

        self.score += FLOOR_PATTERN[:len(self.floor)].sum()
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
        for i in range(NUM_MARKS):
            if np.all(self.grid[i, :] != 0):
                finalScore += PTS_PER_ROW
            if np.all(self.grid[:, i] != 0):
                finalScore += PTS_PER_COL
            if np.sum(self.grid == i+1) == NUM_MARKS:
                finalScore += PTS_PER_SET
        return finalScore


class Table:
    numPlayers = 4

    def __init__(self):
        self.bag = np.random.permutation(
            np.tile(np.arange(1, NUM_MARKS+1), NUM_TOTAL_TILES // NUM_MARKS))
        self.discard = []
        self.player = [Board() for _ in range(self.numPlayers)]
        self.numFactories = NUM_FACTORY_VS_PLAYER[self.numPlayers]
        self.reset()

        self.player[2].grid = np.array([
            [1, 2, 3, 4, 5],
            [0, 1, 0, 3, 4],
            [4, 0, 1, 0, 3],
            [0, 4, 0, 0, 2],
            [0, 3, 4, 0, 0]])
        self.player[2].line = [
            [1], [2, 2], [5, 5], [], [1, 1, 1, 1, 1]
        ]
        self.player[2].floor = [-1, 2, 4]
        self.center = [-1] + [1]*5 + [2]*5 + [3]*5

    def reset(self):
        # reshuffle tiles from discard
        if self.bag.shape[0] < self.numFactories * NUM_MARKS_PER_FACTORY:
            self.bag = np.random.permutation(self.discard)

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

    def step_move(self, mark, factoryIdx, playerIdx, row):
        # step game for a single player move

        factory = self.factory[factoryIdx] if factoryIdx >= 0 else self.center
        player = self.player[playerIdx]
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

    def get_observation(self):
        obs = {
            'factory': self.factory,
            'center': self.center,
            'player': [],
        }
        obs['player'] = [
            {'grid': self.player[i].grid,
             'line': self.player[i].line,
             'score': self.player[i].score,
             'floor': self.player[i].floor} for i in range(self.numPlayers)]
        return obs

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
                playerIdx = (playerIdx + 1) % self.numPlayers

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

    def is_valid(self, mark, factoryIdx, playerIdx, row):
        # TODO
        assert False, "Not implemented"

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


def valid_move(obs, playerIdx, factoryIdx, mark, row):
    #
    player = obs['player'][playerIdx]
    factory = obs['factory'][factoryIdx] if factoryIdx >= 0 else obs['center']
    numMark = factory.count(mark)

    # valid for factory
    if numMark == 0:
        return False

    # valid for player
    if row == -1:
        return True
    return (
        len(player['line'][row]) <= row
        and (len(player['line'][row]) == 0 or player['line'][row][0] == mark)
        and np.all(player['grid'][row] != mark)
    )


def random_move(playerIdx, obs):
    # random valid move
    MAX_TRIES = 1000

    numFactories = len(obs['factory'])
    foundValid = False
    for reps in range(MAX_TRIES):
        factoryIdx = np.random.randint(numFactories + 1) - 1
        mark = np.random.randint(NUM_MARKS) + 1
        if reps < MAX_TRIES // 2:
            row = np.random.randint(NUM_MARKS)
        else:
            row = -1

        foundValid = valid_move(obs, playerIdx, factoryIdx, mark, row)
        if foundValid:
            # print("Found in", reps, "reps")
            return mark, factoryIdx, row

    assert foundValid, f"Player {playerIdx} could not find valid move"
    return None, None, None
