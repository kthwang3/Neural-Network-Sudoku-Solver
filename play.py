import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import *

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(81 * 9))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()


filepath2 = r"C:\Users\pugwe\Desktop\PythonSudoku\best_weights.keras"

# Correct the path to the saved weights
model.load_weights(filepath2)

def solve_sudoku_with_nn(model, puzzle):
    # Preprocess the input Sudoku puzzle
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    initial_board = np.array([int(j) for j in puzzle]).reshape((9, 9, 1))
    initial_board = (initial_board / 9) - 0.5

    while True:
        # Use the neural network to predict values for empty cells
        predictions = model.predict(initial_board.reshape((1, 9, 9, 1))).squeeze()
        pred = np.argmax(predictions, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(predictions, axis=1).reshape((9, 9)), 2)

        initial_board = ((initial_board + 0.5) * 9).reshape((9, 9))
        mask = (initial_board == 0)

        if mask.sum() == 0:
            # Puzzle is solved
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        initial_board[x][y] = val
        initial_board = (initial_board / 9) - 0.5

    # Convert the solved puzzle back to a string representation
    solved_puzzle = ''.join(map(str, initial_board.flatten().astype(int)))

    return solved_puzzle

def print_sudoku_grid(puzzle):
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)

        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(puzzle[i * 9 + j], end=" ")
        print()


def format_sudoku(input_string):
    if len(input_string) != 81 or not input_string.isdigit():
        raise ValueError("Input must be an 81-digit string.")

    formatted_sudoku = ""
    for i in range(9):
        row = " ".join(input_string[i * 9:(i + 1) * 9])
        formatted_sudoku += "    " + row + "\n"

    return formatted_sudoku.strip()

while True:
    print("Welcome to Sudoku Solver, Enter a 81 digit sudoku grid to be solved. ")
    user_puzzle = input()
    formatted_puzzle = format_sudoku(user_puzzle)
    solved_puzzle_nn = solve_sudoku_with_nn(model,formatted_puzzle)
    # Print the solved puzzle as a grid
    print("Sudoku Solution (NN):")
    print_sudoku_grid(solved_puzzle_nn)

    print("Another puzzle? (Y/N)")
    user_again = input()
    if user_again == "Y":
        continue
    else:
        print("Goodbye!")
        break

#solved_puzzle_nn = solve_sudoku_with_nn(model, game)



