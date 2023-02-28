import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from functools import partial

# set seed and some global constants for reproducibility
random_seed = 1
epochs = 10


# Load and clean data
df = pd.read_csv('ewma_df_2016_2022.csv')
df = df.loc[:, df.isnull().mean() < .05]
df = df[df["week"] <= 17]
drop_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'game_date_home'
    , 'game_date_away', 'month', 'day', 'year']
df.drop(drop_cols, axis=1, inplace=True)
# create train and test sets
target = 'home_team_win'
y = pd.DataFrame(df, columns=[target])
X = df.drop([target], axis=1)
# https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=random_seed)

# let's define NN architecture. We'll use batch normalization to help reduce the vanishing/exploding gradient problem.
tf.keras.backend.clear_session()
tf.random.set_seed(random_seed)
model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])

# let's compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics="accuracy")
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))
print('Batch Norm:')
model.evaluate(X_test, y_test)

#
# # Avoiding Overfitting Through Regularization
#
# Let’s try a neural network architecture that uses both L1 and L2 regularization. L1 works to promote a sparse model
# (more weights equal to zero) and L2 to constrain the neural network’s connection weights (makes them smaller).
tf.keras.backend.clear_session()
tf.random.set_seed(random_seed)

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l1_l2(0.1, 0.01))
model = tf.keras.Sequential([
    RegularizedDense(100),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics="accuracy")
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))
print('L1 L2:')
model.evaluate(X_test, y_test)

# Next, we’ll try another regularization technique, dropout.
tf.keras.backend.clear_session()
tf.random.set_seed(random_seed)

model = tf.keras.Sequential([
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=epochs,validation_data=(X_valid, y_valid))

# Training accuracy is lower than the validation accuracy because dropout is only active during training. Evaluate the
# model on the training set after training (i.e., with dropout turned off to get real training accuracy.
print('Dropout train:')
model.evaluate(X_train_full, y_train_full)
print('Dropout test:')
model.evaluate(X_test, y_test)

# All three architectures achieved greater than 50% accuracy on the test set, dropout performing the best at 57% and
# L1, L2 the worst at 53%. We should tune the architectures and hyperparameters to see if we can achieve better
# performance. Further, despite the classification model achieving a greater than 50% score, it should be tested
# against a dummy model such as one that picks the team with the better record. If it has an edge over the dummy model
# then it is providing valuable predictive insight.