#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

DEBUG = False

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    """
    A Tic-Tac-Toe State
    """

    def __init__(self):
        """
        The board is represented by an n * n array,
        >    1 represents X (plays first)
        >    -1 represents O
        >    0 represents an empty position
        """
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    def hash(self):
        """
        Compute the hash value for one state, it's unique
        :return: int
        """
        if self.hash_val is None:
            self.hash_val = 0
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * 3 + i
        return int(self.hash_val)

    def is_end(self):
        """
        Check whether a player has won the game, or it's a tie
        Update the winner and return True, otherwise False
        :return: boolean
        """
        if self.end is not None:
            return self.end

        results = []

        # check row
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(0, BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        # check diagonals
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS - 1 - i]

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum = np.sum(np.abs(self.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    def next_state(self, i, j, symbol):
        """
        Put symbol in position (i, j)
        :param i: x position
        :param j: y position
        :param symbol: -1 for X, 1 for O
        :return: State
        """
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def render(self):
        """
        print the board
        :return: None
        """
        map = {-1: 'O', 1: 'X', 0: '.'}
        print('==========')
        print(''.join(map[i] for i in self.data[0]))
        print(''.join(map[i] for i in self.data[1]))
        print(''.join(map[i] for i in self.data[2]))
        print('==========')


def get_all_states_impl(current_state, current_symbol, all_states):
    """
    Recursively get all possible states
    :param current_state:
    :param current_symbol:
    :param all_states:
    :return:
    """
    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            if current_state.data[i][j] == 0:
                newState = current_state.next_state(i, j, current_symbol)
                newHash = newState.hash()
                if newHash not in all_states.keys():
                    isEnd = newState.is_end()
                    all_states[newHash] = (newState, isEnd)
                    if not isEnd:
                        get_all_states_impl(newState, -current_symbol, all_states)


def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# ALL possible board configurations
all_states = get_all_states()


class Player:
    """
    AI Player
    """

    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        # Winning: reward = 1
        # Losing: reward = 0
        # Ties and all other: reward = 0.5
        for hash_val in all_states.keys():
            (state, is_end) = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    def backup(self):
        """
        Update value estimation
        """
        if DEBUG:
            print('Player Trajectory')
            for state in self.states:
                all_states[state].render()

        self.states = [state.hash() for state in self.states]

        # traverse from back from last state to first state
        # update all the estimation values for states made
        for i in reversed(range(len(self.states) - 1)):
            state_t = self.states[i]
            state_t1 = self.states[i + 1]
            td_error = self.greedy[i] * (self.estimations[state_t1] - self.estimations[state_t])
            self.estimations[state_t] += self.step_size * td_error

    def act(self):
        """
        Choose an action based on the state
        :return: (position, symbol)
        """
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        np.random.shuffle(values)
        # pick the highest estimation value from available positions
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_{}.bin'.format('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_{}.bin'.format('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)


class Judge:
    """
    player1: X (1) play first
    player2: O (-1) play second
    feedback: if True, both players will receive rewards when game is end
    """

    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, print=False):
        """
        :param print: if True, print each board during the game
        :return: the winner symbol
        """
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        while True:
            player = next(alternator)
            if print:
                current_state.render()
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if is_end:
                if print:
                    current_state.render()
                return current_state.winner


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        return

    def reset(self):
        return

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def backup(self, _):
        return

    def act(self):
        self.state.render()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // int(BOARD_COLS)
        j = data % BOARD_COLS
        return i, j, self.symbol


def train(epochs):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judge = Judge(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judge.play(print=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        print('Epoch {}, Player X win {}, Player O win {}'.format(i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()

    player1.save_policy()
    player2.save_policy()


def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judge(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for i in range(0, turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


def play():
    """
    The game is a zero sum game. If both players are playing with an optimal strategy,
    every game will end in a tie. So we test whether the AI can guarantee at least
    a tie if it goes second.
    :return:
    """
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judge = Judge(player1, player2)
        player2.load_policy()
        winner = judge.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    train(int(1e5))
    # compete(int(1e3))
    play()
