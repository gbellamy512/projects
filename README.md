# NFL Projects

## eda_regression_classification
This project walks through Exploratory Data Analysis, (light) Feature Engineering, and Feature Scaling and Transformation. It then trains random forest (using randomized search) and elastic net regression models. Finally, it trains an ensemble classifier that uses random forest, logistic regression, and support vector machine models as its components. 

## deep_neural_network
This project trains and compares the performance of deep neural network classifiers that use different regularization techniques (batch norm, L1/L2, and dropout).

## deep_reinforcement_learning
This one is the most fun. The DRL agent observes a state consisting of down, yards to first, and yards to goal components then chooses to run, pass, or kick a field goal. Deep Q-Network (DQN), Fixed DQN, and Double DQN methodologies are tested. (Note: the project is functional but needs to be cleaned up to increase readability and further tuning to become more effective).

## References
### All Projects
[Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems 3rd Edition](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975/ref=sr_1_1?crid=XNK7ONTN4KHR&keywords=hands+on+machine+learning+with+scikit-learn+and+tensorflow+3&qid=1677762985&sprefix=hands+on+machine+%2Caps%2C98&sr=8-1) by Aurélien Géron
### Deep Reinforcement Learning
Nicholas Renotte's '[Building a Custom Environment for Deep Reinforcement Learning with OpenAI Gym and Python](https://www.youtube.com/watch?v=bD6V3rcr_54&list=PLgNJO2hghbmjlE6cuKMws2ejC54BTAaWV&index=5)' youtube video

[Grokking Deep Reinforcement Learning 1st Edition](https://www.amazon.com/Grokking-Reinforcement-Learning-Miguel-Morales/dp/1617295450/ref=sr_1_8?crid=64IW2E7YY7BW&keywords=reinforcement+learning&qid=1677549970&sprefix=reinforc%2Caps%2C128&sr=8-8) by Miguel Morales

## Data
ewma_df_2016_2022 - Contains Estimated Points Added (EPA) broken down by team and play type. For more information on EPA and how to source the data check out Ben Dominguez’s awesome '[Build an NFL win probability model (Super Bowl Sunday Edition)](https://www.fantasyfootballdatapros.com/blog/ff-analysis/6)' blog post.
