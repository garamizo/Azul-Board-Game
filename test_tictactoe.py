from time import sleep
from random import choice
from azul.tictactoe import reset_round, print_state, get_action_space, is_action_valid, is_terminal, play, get_reward
from azul.ai import maxn, paranoid, MCTS_node
from copy import deepcopy

if __name__ == "__main__":
    # player 0 is human
    print("Input q to quit")
    state = reset_round()
    root = MCTS_node(deepcopy(state))
    # root.grow(timeout=0.1)
    isDone = False
    while not isDone:
        if state['activePlayer'] == 0:
            print_state(state)
            inp = input("Input action for player 0 (row collumn): ")
            if inp == 'q':
                quit()
            action = [int(s) for s in inp.split(" ")]
        else:
            newRoot = MCTS_node.search_node(root, state)
            if newRoot == None:
                print("Creating new search tree")
                root = MCTS_node(deepcopy(state))
            else:
                root = newRoot
                root.parent = None

            root.grow(timeout=3.0)
            action = root.get_best_action()[0]
            # action = paranoid(state, 7, -100, 100, state['activePlayer'])[1]
            print(f"AI player {state['activePlayer']} acts: {action}")
            sleep(1)

        assert is_action_valid(state, action), "Invalid action"
        play(state, action)

        if is_terminal(state):
            reward = get_reward(state)
            winners = [i for i, v in enumerate(reward) if v == max(reward)]
            print(f"\nGame Over\nPlayer {winners} wins\n")

            state = reset_round(state)
            sleep(1)
