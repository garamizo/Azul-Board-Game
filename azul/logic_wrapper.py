import pythonnet
pythonnet.load("coreclr")  # to use dotnet rather than mono

if True:
    import clr
    import sys
    sys.path.append(r"AzulLibrary/bin/Release/net6.0")
    clr.AddReference('AzulLibrary')  # add .dll file
    from Azul import Game, GameAction
    from Ai import MCTS

NUM_ROWS = 5
NUM_COLS = 5
NUM_COLORS = 5


class Table:

    def __init__(self, numPlayers=3, game=None):
        self.core = Game(numPlayers) if game is None else game

    @property
    def numFactories(self):
        return self.core.numFactories

    @property
    def activePlayer(self):
        return self.core.activePlayer

    @property
    def numPlayers(self):
        return self.core.numPlayers

    def step_move(self, color, factoryIdx, row):
        assert color > 0 and color <= NUM_COLORS, "Error"
        self.core.Play(
            GameAction(self.numFactories if factoryIdx < 0 else factoryIdx,
                       color-1, NUM_ROWS if row < 0 else row, self.core))

    def get_printable(self):
        return Game.ToScript(self.core)

    def get_observation(self):
        a = self.core

        def column_to_color(row, col):
            return (col - row) % NUM_COLORS + 1

        factory = []
        for factoryi in a.factories:
            fac = []
            for color, count in enumerate(factoryi):
                fac += [color+1 if color < NUM_COLORS else -1]*count
            factory.append(fac)
        center, factory = factory[-1], factory[:-1]

        players = []
        for p in a.players:
            grid = []
            for i in range(NUM_ROWS):
                row = []
                for j in range(NUM_COLS):
                    row.append(p.grid[i, j] * column_to_color(i, j))
                grid.append(row)

            floor = [-1] if p.floor[NUM_COLORS] > 0 else []
            for color in range(NUM_COLORS):
                floor += [color+1] * p.floor[color]

            lines = []
            for row in range(NUM_ROWS):
                line = []
                for color in range(NUM_COLORS):
                    line += [color+1] * p.line[row, color]
                lines.append(line)

            player = dict(grid=grid, line=lines, score=p.score, floor=floor)
            players.append(player)

        bag, discard = [], []
        for color in range(NUM_COLORS):
            bag.append(a.bag[color])
            discard.append(a.discarded[color])

        return dict(factory=factory, center=center, player=players, activePlayer=a.activePlayer, roundIdx=a.roundIdx, bag=bag, discard=discard)

    def is_valid(self, color, factoryIdx, row):
        # center is factory 0 on C#
        # factoryIdx for center is -1 on python
        # floor is -1 on python, NUM_ROWS in C#
        return self.core.IsValid(self.numFactories if factoryIdx < 0 else factoryIdx,
                                 color-1, NUM_ROWS if row < 0 else row)

    def is_game_over(self):
        return self.core.IsGameOver()

    @staticmethod
    def get_action(a, gameCore):
        # convert from core to python
        return -1 if a.factoryIdx >= gameCore.numFactories else a.factoryIdx, \
            a.color+1, -1 if a.row >= NUM_ROWS else a.row
