# NFL Projects

## eda_regression_classification
This project walks through Exploratory Data Analysis, (light) Feature Engineering, and Feature Scaling and Transformation. It then trains random forest (using randomized search) and elastic net regression models. Finally, it trains an ensemble classifier that uses random forest, logistic regression, and support vector machine models as its components. 

## deep_neural_network
This project trains and compares the performance of deep neural network classifiers that use different regularization techniques (batch norm, L1/L2, and dropout).

## deep_reinforcement_learning
This one is the most fun. The DRL agent observes a state consisting of down, yards to first, and yards to goal components then chooses to run, pass, kick a field goal, or punt. Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms were tested. Stable Baselines3 was used to train the agents, and Weights and Biases was used to monitor and evaluate the training runs.

## Data
ewma_df_2016_2022 - Contains Estimated Points Added (EPA) broken down by team and play type. For more information on EPA and how to source the data check out Ben Dominguez’s awesome '[Build an NFL win probability model (Super Bowl Sunday Edition)](https://www.fantasyfootballdatapros.com/blog/ff-analysis/6)' blog post.
