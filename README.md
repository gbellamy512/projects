# NFL Projects

## eda_regression_classification
This project walks through Exploratory Data Analysis, (light) Feature Engineering and Feature Scaling and Transformations. It then trains a random forest regression model using randomized search and an elastic net regression model. Finally, we train an ensemble classifier that uses three different model types. 

## deep_neural_network
This project trains and compares different deep neural network classifiers using batch normalization, L1/L2 regularization, and dropout techniques.

## deep_reinforcement_learning
This one is the most fun. It works but needs  cleanup to be more readable and further tuning to become more effective. The DRL model takes a down, yards to first, and yards to goal state and then choices an action (run, pass, or kick a field goal). The agent must balance trying to score with avoiding giving the ball to their opponent with good field position. Deep Q-Network (DQN), Fixed DQN, and Double DQN methodologies are tested.

## References
All projects rely heavily on ‘the Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems 3rd Edition’ - https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975/ref=sr_1_1?crid=BN1JBCP8AT2W&keywords=hands+on+machine+learning+with+scikit-learn+and+tensorflow+3&qid=1677549921&sprefix=hands+on+machine+le%2Caps%2C110&sr=8-1 
The reinforcement learning project also leans on ‘Grokking Deep Reinforcement Learning 1st Edition’ - https://www.amazon.com/Grokking-Reinforcement-Learning-Miguel-Morales/dp/1617295450/ref=sr_1_8?crid=64IW2E7YY7BW&keywords=reinforcement+learning&qid=1677549970&sprefix=reinforc%2Caps%2C128&sr=8-8 and Nicholas Renotte’s ‘Building a Custom Environment for Deep Reinforcement Learning with OpenAI Gym and Python’ video - https://www.youtube.com/watch?v=bD6V3rcr_54&list=PLgNJO2hghbmjlE6cuKMws2ejC54BTAaWV&index=4.

## Data
These projects use Estimated Points Added (EPA) data to build regression and classification models. See this awesome article about more info on EPA and how to derive a dataset - https://www.fantasyfootballdatapros.com/blog/ff-analysis/6.
