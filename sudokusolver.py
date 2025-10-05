import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import *

data = pd.read_csv(r"C:\Users\pugwe\Desktop\PythonSudoku\sudoku.csv")
try:
    data = pd.DataFrame({"quizzes": data["puzzle"], "solutions": data["solution"]})
except KeyError:
    pass

class DataGenerator(Sequence):
    def __init__(self, df, batch_size=16, subset="train", shuffle=False, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.info = info
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 9, 9, 1))
        y = np.empty((self.batch_size, 81, 1))
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        for i, f in enumerate(self.df['quizzes'].iloc[indexes]):
            self.info[index * self.batch_size + i] = f
            X[i,] = (np.array(list(map(int, list(f)))).reshape((9, 9, 1)) / 9) - 0.5
        if self.subset == 'train':
            for i, f in enumerate(self.df['solutions'].iloc[indexes]):
                self.info[index * self.batch_size + i] = f
                y[i,] = np.array(list(map(int, list(f)))).reshape((81, 1)) - 1
        if self.subset == 'train': 
            return X, y
        else: 
            return X

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

train_idx = int(len(data) * 0.95)
data = data.sample(frac=1).reset_index(drop=True)
training_generator = DataGenerator(data.iloc[:train_idx], subset="train", batch_size=640)
validation_generator = DataGenerator(data.iloc[train_idx:], subset="train", batch_size=640)

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

filepath1 = r"C:\Users\pugwe\Desktop\PythonSudoku\weights-improvement-{epoch:02d}-{val_accuracy:.2f}.keras"
filepath2 = r"C:\Users\pugwe\Desktop\PythonSudoku\best_weights.keras"
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    verbose=1,
    min_lr=1e-6
)
callbacks_list = [checkpoint1, checkpoint2, reduce_lr]

history = model.fit(training_generator, validation_data=validation_generator, epochs=5, verbose=1, callbacks=callbacks_list)

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

new_game = '''
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
'''

game = '''
    0 0 0 3 0 0 0 6 0
    0 6 4 0 0 1 0 2 0
    0 0 0 5 0 0 0 0 0
    0 2 3 0 5 0 6 0 0
    0 0 9 8 0 3 7 0 0 
    0 0 8 0 9 0 1 4 0
    0 0 0 0 0 9 0 0 0 
    0 1 0 4 0 0 8 9 0 
    0 5 0 0 0 7 0 0 0 
'''

solved_puzzle_nn = solve_sudoku_with_nn(model, game)

# Print the solved puzzle as a grid
print("Sudoku Solution (NN):")
print_sudoku_grid(solved_puzzle_nn)

