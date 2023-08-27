import numpy as np
from random import choice
from time import sleep

NUM_ROWS, NUM_COLS, NUM_SET = 4, 4, 3
NUM_PLAYERS = 3
EMPTY = -1


def is_action_valid(state, action):
    row, col = action
    return state['grid'][row][col] == EMPTY


def reset_round(state=None, startingPlayer=0):
    # create new game
    if state is None:
        state = {
            'grid': (np.ones([NUM_ROWS, NUM_COLS], int) * EMPTY).tolist(),
            'activePlayer': startingPlayer,
            'numPlayers': NUM_PLAYERS,
            'startingPlayer': startingPlayer,
            'roundIdx': 0,
        }
    else:
        numPlayers = state['numPlayers']
        state['roundIdx'] += 1
        state['startingPlayer'] = (
            state['startingPlayer'] + 1) % state['numPlayers']
        state['activePlayer'] = state['startingPlayer']
        state['grid'] = (np.ones([NUM_ROWS, NUM_COLS], int) * EMPTY).tolist()

    return state


def is_terminal(state):
    """
        If game reached end of round ie. factories and center are empty
        Does not consider other rounds
    """
    def count(rowDir, colDir):
        for i in range(1, NUM_SET + 1):
            r, c = row + i * rowDir, col + i * colDir
            if (
                r < 0
                or c < 0
                or r >= NUM_ROWS
                or c >= NUM_COLS
                or state['grid'][r][c] != p
            ):
                return i-1
        return NUM_SET

    for p in range(state['numPlayers']):
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                if (state['grid'][row][col] == p and (
                        count(1, 0) + count(-1, 0) + 1 >= NUM_SET
                        or count(0, 1) + count(0, -1) + 1 >= NUM_SET
                        or count(1, 1) + count(-1, -1) + 1 >= NUM_SET
                        or count(1, -1) + count(-1, 1) + 1 >= NUM_SET)
                        ):
                    return True

    numEmpties = sum([gRow.count(EMPTY) for gRow in state['grid']])
    # print(f"{numEmpties=}")
    return numEmpties == 0


def action_ends_game(state, action):
    # call after is_round_over, true if a player has a completed row
    def count(rowDir, colDir):
        for i in range(1, NUM_SET + 1):
            r, c = row + i * rowDir, col + i * colDir
            if (
                r < 0
                or c < 0
                or r >= NUM_ROWS
                or c >= NUM_COLS
                or state['grid'][r][c] != p
            ):
                return i-1
        return NUM_SET

    row, col = action
    p = state['activePlayer']
    return (
        count(1, 0) + count(-1, 0) + 1 >= NUM_SET
        or count(0, 1) + count(0, -1) + 1 >= NUM_SET
        or count(1, 1) + count(-1, -1) + 1 >= NUM_SET
        or count(1, -1) + count(-1, 1) + 1 >= NUM_SET
    )


REWARD_SUM_MAX = 0
MAX_NUM_ACTIONS = NUM_COLS * NUM_ROWS
MIN_NUM_ACTIONS = NUM_SET + (NUM_SET - 1) * (NUM_PLAYERS - 1)  # 3 + 2 * 1


def interp(x, x1, y1, x2, y2):
    return (x - x1) * (y2 - y1) / (x2 - x1) + y1


def get_reward(state):
    """
        If game reached end of round ie. factories and center are empty
        Does not consider other rounds
    """
    def count(rowDir, colDir):
        for i in range(1, NUM_SET + 1):
            r, c = row + i * rowDir, col + i * colDir
            if (
                r < 0
                or c < 0
                or r >= NUM_ROWS
                or c >= NUM_COLS
                or state['grid'][r][c] != p
            ):
                return i-1
        return NUM_SET

    reward = [1 / NUM_PLAYERS] * state['numPlayers']
    for p in range(state['numPlayers']):
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                if (state['grid'][row][col] == p and (
                        count(1, 0) + count(-1, 0) + 1 >= NUM_SET
                        or count(0, 1) + count(0, -1) + 1 >= NUM_SET
                        or count(1, 1) + count(-1, -1) + 1 >= NUM_SET
                        or count(1, -1) + count(-1, 1) + 1 >= NUM_SET)
                        ):
                    numActions = NUM_COLS * NUM_ROWS - \
                        sum([gRow.count(EMPTY) for gRow in state['grid']])
                    winReward = interp(numActions, MIN_NUM_ACTIONS,
                                       1, MAX_NUM_ACTIONS + 1, 1 / NUM_PLAYERS)
                    loseReward = (1 - winReward) / (NUM_PLAYERS - 1)
                    reward = [winReward if p == i else
                              loseReward for i in range(state['numPlayers'])]
                    # reward = [1 if p == i else
                    #           0 for i in range(state['numPlayers'])]
                    return reward

    return reward


def get_reward_simple(state):
    return get_reward(state)


def get_action_space(state):
    """
        Return:
            ~List of dict [{'factoryIdx': 0, 'colorIdx': 0, 'row': 0}]~
            List of list
    """
    actions = []
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            if state['grid'][row][col] == EMPTY:
                actions.append([row, col])
    return actions


def play(state, action):
    """"
        Update game state given action
        Does not update score
        No returns
        ends at round, not end of game
    """
    row, col = action
    state['grid'][row][col] = state['activePlayer']
    state['activePlayer'] = (state['activePlayer'] + 1) % state['numPlayers']


def print_state(s):
    activePlayer, numPlayers, startingPlayer, roundIdx = s[
        'activePlayer'], s['numPlayers'], s['startingPlayer'], s['roundIdx']
    print(f"{activePlayer=}\t{numPlayers=}\t{startingPlayer=}\t{roundIdx=}")
    txt = str()
    for gRow in s['grid']:
        txt += str(" ")
        for val in gRow:
            txt += str(val if val != EMPTY else " ") + str(" | ")
        txt = txt[:-3:]
        txt += "\n" + "---+" * (NUM_COLS - 1) + "---\n"
    txt = txt[:-(NUM_COLS*4 + 1)]
    print(txt)


if __name__ == "__main__":
    # player 0 is human
    print("Input q to quit")
    state = reset_round()
    isDone = False
    while not isDone:
        if state['activePlayer'] == 0:
            print_state(state)
            inp = input("Input action (row collumn): ")
            if inp == 'q':
                quit()
            action = [int(s) for s in inp.split(" ")]
        else:
            action = choice(get_action_space(state))
            print(f"\nAI player acts: {action}")
            sleep(1)

        assert is_action_valid(state, action), "Invalid action"
        play(state, action)

        if is_terminal(state):
            winner = (state['activePlayer'] - 1) % state['numPlayers']
            print(f"\nGame Over\nPlayer {winner} wins\n")
            state = reset_round(state)
            sleep(1)
