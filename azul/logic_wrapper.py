import pythonnet
pythonnet.load("coreclr")  # to use dotnet rather than mono

if True:
    import clr
    import sys
    sys.path.append(r"AzulLibrary/bin/Release/net7.0")
    clr.AddReference('AzulLibrary')  # add .dll file
    # from Azul import Game, Move
    from Azul import Game, Move
    # from Ai import MCTS

NUM_ROWS = 5
NUM_COLS = 5
NUM_COLORS = 5

COLOR_NAME = {0: "BLUE", 1: "YELLOW", 2: "RED",
              3: "BLACK", 4: "WHITE", 5: "FIRST"}
ROW_NAME = {-1: "FLOOR", 0: "FIRST", 1: "SECOND",
            2: "THIRD", 3: "FOURTH", 4: "FIFTH"}


# class MoveUI(Move):
#     rowOn = False
#     factoryIdxOn = False
#     colorOn = False
#     gridOn = [False] * 25
#     # isRegularPhase = False

#     def __init__(self, *argv):
#         # if len(argv) == 1:
#         pass

#     def is_set(self):
#         if self.isRegularPhase:
#             return self.colorOn and self.factoryIdxOn and self.rowOn
#         else:
#             return not any(self.gridOn)

#     @property
#     def isRegularPhase(self):
#         return self.colIdx[0] == -2

#     @isRegularPhase.setter
#     def isRegularPhase(self, value):
#         # on transition, reset all
#         # if (self.isRegularPhase == False and value == True) or \
#         #    (self.isRegularPhase == True and value == False):
#         #     self.colorOn = False
#         #     self.factoryIdxOn = False
#         #     self.rowOn = False
#         #     self.gridOn = [False] * 25
#         # self.isRegularPhase = value
#         if value:
#             self.colIdx[0] = -2


def get_observation(a):
    # def column_to_color(row, col):
    #     return (col - row) % NUM_COLORS + 1

    factory = []
    for factoryi in a.factories:
        fac = []
        for color, count in enumerate(factoryi):
            fac += [color]*count
        factory.append(fac)
    center, factory = factory[-1], factory[:-1]

    players = []
    for p in a.players:
        grid = []
        for i in range(NUM_ROWS):
            row = []
            for j in range(NUM_COLS):
                row.append(p.grid[i, j])
            grid.append(row)

        floor = []
        for color in range(NUM_COLORS + 1):
            floor += [color] * p.floor[color]

        lines = []
        for row in range(NUM_ROWS):
            line = []
            for color in range(NUM_COLORS):
                line += [color] * p.line[row, color]
            lines.append(line)

        player = dict(grid=grid, line=lines, score=p.score, floor=floor)
        players.append(player)

    bag, discard = [], []
    for color in range(NUM_COLORS):
        bag.append(a.bag[color])
        discard.append(a.discarded[color])

    return dict(factory=factory, center=center, player=players,
                activePlayer=a.activePlayer, roundIdx=a.roundIdx,
                bag=bag, discard=discard)


# class Table:

#     def __init__(self, numPlayers=3, game=None):
#         self.core = Game(numPlayers) if game is None else game

#     @property
#     def numFactories(self):
#         return self.core.numFactories

#     @property
#     def activePlayer(self):
#         return self.core.activePlayer

#     @property
#     def numPlayers(self):
#         return self.core.numPlayers

#     def step_move(self, move):
#         # assert color > 0 and color <= NUM_COLORS+1, "Error"
#         self.core.Play(move)

#     def get_printable(self):
#         return Game.ToScript(self.core)

#     def get_observation(self):
#         a = self.core

#         def column_to_color(row, col):
#             return (col - row) % NUM_COLORS + 1

#         factory = []
#         for factoryi in a.factories:
#             fac = []
#             for color, count in enumerate(factoryi):
#                 fac += [color+1 if color < NUM_COLORS else -1]*count
#             factory.append(fac)
#         center, factory = factory[-1], factory[:-1]

#         players = []
#         for p in a.players:
#             grid = []
#             for i in range(NUM_ROWS):
#                 row = []
#                 for j in range(NUM_COLS):
#                     row.append(p.grid[i, j] * column_to_color(i, j))
#                 grid.append(row)

#             floor = [-1] if p.floor[NUM_COLORS] > 0 else []
#             for color in range(NUM_COLORS):
#                 floor += [color+1] * p.floor[color]

#             lines = []
#             for row in range(NUM_ROWS):
#                 line = []
#                 for color in range(NUM_COLORS):
#                     line += [color+1] * p.line[row, color]
#                 lines.append(line)

#             player = dict(grid=grid, line=lines, score=p.score, floor=floor)
#             players.append(player)

#         bag, discard = [], []
#         for color in range(NUM_COLORS):
#             bag.append(a.bag[color])
#             discard.append(a.discarded[color])

#         return dict(factory=factory, center=center, player=players, activePlayer=a.activePlayer, roundIdx=a.roundIdx, bag=bag, discard=discard)

#     def is_valid(self, move):
#         # center is factory 0 on C#
#         # factoryIdx for center is -1 on python
#         # floor is -1 on python, NUM_ROWS in C#
#         if move.isRegularPhase == False:
#             return self.core.IsValid(move.colIdx, self.core.activePlayer)
#         else:
#             # return self.core.IsValid(self.numFactories if move.factoryIdx < 0 else move.factoryIdx,
#             #                          move.color-1, NUM_ROWS if move.row < 0 else move.row)
#             return self.core.IsValid(move.factoryIdx, move.color, move.row)

#     def is_game_over(self):
#         return self.core.IsGameOver()

#     def get_egreedy_move(self, epsilon=0.1):
#         return Table.get_action(self.core.GetEGreedyMove(epsilon), self.core)

#     @staticmethod
#     def get_action(a, gameCore):
#         # convert from core to python
#         if a.isRegularPhase:
#             return Move(-1 if a.factoryIdx >= gameCore.numFactories else a.factoryIdx,
#                         a.color+1, -1 if a.row >= NUM_ROWS else a.row)
#         else:
#             return Move(a.colIdx)
#         # return -1 if a.factoryIdx >= gameCore.numFactories else a.factoryIdx, \
#         #     a.color+1, -1 if a.row >= NUM_ROWS else a.row


# class Move:
#     color = None
#     factoryIdx = None
#     row = None
#     colIdx = [None] * 5
#     # colors = None
#     # isFirst = None
#     isRegularPhase = None

#     def __init__(self, *argv):
#         if len(argv) == 1:  # isRegularPhase or colIdx
#             try:
#                 iter(argv[0])
#                 self.colIdx = argv[0]
#                 self.isRegularPhase = False
#             except TypeError:
#                 self.isRegularPhase = argv[0]
#         elif len(argv) == 3:  # factoryIdx, color, row
#             self.factoryIdx = argv[0]
#             self.color = argv[1]
#             self.row = argv[2]
#             self.isRegularPhase = True
#         else:
#             raise Exception("Error")

#     def is_selected(self):
#         if self.isRegularPhase is None:
#             return False
#         if self.isRegularPhase:
#             return self.color is not None and self.factoryIdx is not None and self.row is not None
#         else:
#             return not any(m is None for m in self.move.colIdx)

#     def __repr__(self):
#         if self.isRegularPhase:
#             return (f"Move({self.factoryIdx}, {COLOR_NAME[self.color]}:{self.color}, "
#                     f"{ROW_NAME[self.row]}:{self.row})")
#         else:
#             return f"Move({self.colIdx})"
