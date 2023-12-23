import warnings
# from logic_wrapper import Table
import pythonnet
pythonnet.load("coreclr")  # to use dotnet rather than mono

if True:
    import clr
    import sys
    sys.path.append(r"AzulLibrary/bin/Release/net7.0")
    clr.AddReference('AzulLibrary')  # add .dll file
    from Azul import Game, Move
    from Ai import MCTS_Stochastic as MCTS
    # from Ai import MCTS
    # why can't I import this module?
    # a) it's not in the same directory
    # q) but it worked for Azul and Ai
    # a) because they are in the same directory
    # q) they are not
    # a) yes they are
    # q) they are on AzulLibrary/
    # from GameUtils import RewardMap
    # from Ai import MCTS


# class MCTS(CMCTS[Game, Move]):
#     def __init__(self, *args):
#         # super().__init__(*args)
#         pass

#     def get_action(self):
#         actionIdx = self.GetBestActionIdx()
#         return self.core.actions[actionIdx], \
#             NumRolls(actionIdx), WinRatio(actionIdx)

#     def SearchNode(self, *args):
#         return MCTS(super().SearchNode(*args).Item1)


# class MCTS_node:

#     def __init__(self, state, parent=None, actionIdx=None):
#         # only used for construction
#         # self.core = MCTS(state.core, parent.core if parent else None)
#         if parent == None:  # root node
#             self.core = MCTS[Game, Move](state.core, 0.05)
#         else:
#             self.core = MCTS[Game, Move](parent.core, actionIdx)

#     @property
#     def numRolls(self):
#         return self.core.numRolls

#     def grow_while(self, timeout=2.0, maxRolls=1_000_000):
#         self.core.GrowWhile(timeout, maxRolls)

#     def get_best_action(self):
#         return self.core.GetBestActionIdx()

#     def get_action(self, actionIdx=None):
#         if actionIdx == None:
#             actionIdx = self.get_best_action()
#         action = self.core.actions[actionIdx]
#         numRolls, numWins = 0, 0
#         for child in self.core.childs[actionIdx]:
#             numRolls += child.numRolls
#             numWins += child.numWins
#         # child = self.core.childs[actionIdx][0]
#         if numWins < 0.001 or numWins > 0.999:
#             return *Table.get_action(self.core.state.GetGreedyMove(), self.core.state), \
#                 numRolls, numWins
#         else:
#             return *Table.get_action(action, self.core.state), \
#                 numRolls, numWins

#         # return *Table.get_action(self.core.state.GetRandomMove(), self.core.state), \
#         #     10_000, 5_000
#         # return *Table.get_action(self.core.state.GetGreedyMove(), self.core.state), \
#         #     10_000, 5_000

#     def search_node(self, state, maxDepth=4):
#         result = self.core.SearchNode(state.core, maxDepth)
#         self.core, found = result.Item1, result.Item2
#         if not found:
#             warnings.warn("Node not found. Creating new tree...")
#         return self
