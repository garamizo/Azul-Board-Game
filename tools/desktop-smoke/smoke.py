"""Plays one greedy game through the desktop app's pythonnet wrappers."""
import sys

sys.path.insert(0, "azul")

from logic_wrapper import Game, Move  # noqa: E402  loads AzulLibrary/bin/Release/net10.0
from ai_wrapper import MCTS  # noqa: E402

game = Game(3)
tree = MCTS[Game, Move](game, 0.0)
tree.GrowWhile(0.2, 500)
assert game.IsValid(tree.GetBestAction()), "MCTS proposed an invalid move"

steps = 0
while not game.IsGameOver():
    move = game.GetGreedyMove()
    assert game.IsValid(move), f"invalid greedy move {move}"
    game.Play(move)
    steps += 1
    assert steps < 2000, "game did not end"

print("desktop smoke ok", steps, [p.score for p in game.players])
