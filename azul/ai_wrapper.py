from logic_wrapper import Table
import pythonnet
pythonnet.load("coreclr")  # to use dotnet rather than mono

if True:
    import clr
    import sys
    sys.path.append(r"AzulLibrary/bin/Release/net6.0")
    clr.AddReference('AzulLibrary')  # add .dll file
    from Azul import Game, GameAction
    from Ai import MCTS_Stochastic as MCTS
    # from Ai import MCTS


class MCTS_node:

    def __init__(self, state, parent=None, actionIdx=None):
        # only used for construction
        self.core = MCTS(state.core, parent.core if parent else None)
        if parent == None:
            self.core = MCTS(state.core)
        else:
            self.core = MCTS(parent.core, actionIdx)

    @property
    def numRolls(self):
        return self.core.numRolls

    def grow_while(self, timeout=2.0, maxRolls=1_000_000):
        self.core.GrowWhile(timeout, maxRolls)

    def get_best_action(self):
        return self.core.GetBestActionIdx()

    def get_action(self, actionIdx):
        action = self.core.actions[actionIdx]
        child = self.core.childs[actionIdx]
        return *Table.get_action(action, self.core.state), \
            child.numRolls, child.numWins

    def search_node(self, state, maxDepth=4):
        self.core = self.core.SearchNode(state.core, maxDepth)
        return self if self.core is not None else None
