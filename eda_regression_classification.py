import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC

# set seed for reproducibility
random_seed = 1

#
## Download the data
#
df = pd.read_csv('ewma_df_2016_2022.csv')
# Calculate the home team margin of victory (negative is a loss) which will be used as the regression target.

#
## Clean the data
#
# Drop columns with >=5% null
# https://stackoverflow.com/questions/43311555/how-to-drop-column-according-to-nan-percentage-for-dataframe
df = df.loc[:, df.isnull().mean() < .05]
print(df.head())
df.info()
# Make sure all 32 teams are present and equally represented.
print('team count:', len(df["home_team"].value_counts()))
print(df["home_team"].value_counts())
# All 32 teams are present but some teams have more occurrences than others. This is due to more playoff appearances.
# Let's see if filtering out weeks 17 and later better balances the distribution across teams.
df = df[df["week"] <= 17]
print(df["home_team"].value_counts())
# Looks better!

#
## Exploratory Data Analysis
#
# Let's take a look at the distributions of our numerical variables by plotting their histograms.
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
df.hist(bins=50, figsize=(12, 8))
plt.show()

# Let's set the target and check how variables correlate with it.
df['home_margin'] = df['home_score'] - df['away_score']
target = 'home_margin'
corr_matrix = df.corr()
print(corr_matrix[target].sort_values(ascending=False))
# Unsurprisingly, passing_offense_home is most correlated with the home team win margin (although still not a strong
# correlation at only ~.2). It’s not surprising that rushing defense has the lowest correlation with winning margin
# but it is that nearly no correlation exists (<.01).
# Let's plot some of the passing correlations to see them visually.
attributes = ["home_margin", "ewma_dynamic_window_passing_offense_home", "ewma_dynamic_window_passing_defense_home",
              "ewma_dynamic_window_passing_offense_away", "ewma_dynamic_window_passing_defense_away"]
scatter_matrix(df[attributes], figsize=(12, 8))
plt.show()
df.plot(kind="scatter", x="ewma_dynamic_window_passing_offense_home", y="home_margin", alpha=0.1, grid=True)
plt.show()

#
## Feature Engineering
#
# Let's create a season categorical variable b/c scoring may decrease in the winter when snow is more likely and (to be
# honest mostly because) we want to have a categorical variable to work with.
# https://stackoverflow.com/questions/60285557/extract-seasons-from-datetime-pandas
df['game_date_home']= pd.to_datetime(df['game_date_home'])
df['date_offset'] = (df.game_date_home.dt.month*100 + df.game_date_home.dt.day - 320)%1300
df['season_weather'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                      labels=['spring', 'summer', 'autumn', 'winter'])
# Above creates dtype 'category' which doesn't work with categorical_features & numeric_features creation logic below.
df['season_weather'] = df['season_weather'].astype(str)

#
## Create train and test sets
#
# drop team, date, and score/win columns that provide no predictive value or the answers
drop_cols = ['date_offset', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_team_win'
    , 'game_date_home', 'game_date_away', 'month', 'day', 'year']
df.drop(drop_cols, axis=1, inplace=True)
y = pd.DataFrame(df, columns=[target])
X = df.drop([target], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

#
# # Feature Scaling and Transformers
#
# https://stackoverflow.com/questions/61641852/what-is-the-valid-specification-of-the-columns-needed-for-sklearn-classifier-p
categorical_features = X.select_dtypes(include="object").columns
numeric_features = X.select_dtypes(exclude="object").columns

# https://adhikary.net/2019/03/23/categorical-and-numeric-data-in-scikit-learn-pipelines/
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
preprocessing = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#
# # Create Pipeline and Train a Random Forest Regressor using random search
#
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=random_seed)),
])

# Random Forest default parameters - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
param_distribs = {
    # default = 100
    'random_forest__n_estimators': randint(low=50, high=500)
    # default = 1
    , 'random_forest__max_features': randint(low=1, high=50)
    # default = None
    , 'random_forest__max_depth': randint(low=1, high=50)
    # default = 2
    , 'random_forest__min_samples_split': randint(low=2, high=100)
    # default = 1
    , 'random_forest__min_samples_leaf': randint(low=1, high=50)
}
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=random_seed)
# https://stackoverflow.com/questions/58313842/a-column-vector-y-was-passed-when-a-1d-array-was-expected-error-message
rnd_search.fit(X_train, np.ravel(y_train))
y_pred = rnd_search.predict(X_test)
print('rs mse: {mse}'.format(mse=mean_squared_error(y_test, y_pred)))
print('rs root of mse: {mse}'.format(mse=math.sqrt(mean_squared_error(y_test, y_pred))))

# let's check feature importance
final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
importance_tuples = sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
           reverse=True)
for (score, name) in importance_tuples:
    print(round(score, 2), name)

# the below save the results in case you want to look into individual errors in excel format
X_test['y'] = y_test
X_test['pred'] = y_pred
X_test['error'] = X_test['pred'] - X_test['y']
X_test.to_csv('rfr.csv')

# The Random Forest Regressor model has about ~13.5 avg error. That is bad in this context.
# Let's try Elastic Net Regression to see if if it works better.

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("Elastic_Net", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_seed)),
])

full_pipeline.fit(X_train, np.ravel(y_train))
y_pred = full_pipeline.predict(X_test)
print('rs mse: {mse}'.format(mse=mean_squared_error(y_test, y_pred)))
print('rs root of mse: {mse}'.format(mse=math.sqrt(mean_squared_error(y_test, y_pred))))

X_test['y'] = y_test
X_test['pred'] = y_pred
X_test['error'] = X_test['pred'] - X_test['y']
X_test.to_csv('enr.csv')

# Again ~13.5 avg error. We could try other regression models, more hyperparameter tuning, and feature engineering but
# it doesn’t seem like these inputs will create a quality victory margin regression model. Let’s see if they work better
# for classification.

#
# # Classification
#

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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

#
# # Create and train model
#
# Ensemble learning trains a group of predictors and uses their combined outputs to derive a final output.
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=random_seed)),
        ('rf', RandomForestClassifier(random_state=random_seed)),
        ('svc', SVC(random_state=random_seed))
    ]
)
voting_clf.fit(X_train, y_train)

# Let's check the score of each individual predictor
for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test,y_test))
# All perform similarly (>61% and <63%) with logistic regression performing the best.
# hard voting averages the ensemble outputs
print('hard voting score:', voting_clf.score(X_test, y_test))
# soft voting picks the estimator with the highest probability
voting_clf.voting = 'soft'
voting_clf.named_estimators['svc'].probability = True
voting_clf.fit(X_train, y_train)
print('soft voting score:', voting_clf.score(X_test, y_test))

# The ensemble slightly outperforms any individual predictor with both hard and soft voting but performs best with hard
# voting, achieving an ~63% score.

#
# # Summary:
#

# We tried building a regression model using both Random Forest and Elastic Net architectures but neither produced a
# reliable output for the home team margin of victory. We could try other regression architectures, additional
# hyperparameter tuning, and feature engineering but it doesn’t seem like the EPA data currently available is well
# suited to create a quality victory margin regression model.

# We next tried classification to predict the winning team as opposed to the margin of victory. We used ensemble
# learning and the result was much more reliable than that of regression. Despite the classification model achieving
# a greater than 50% score, it should be tested against a dummy model such as one that picks the team with the better
# record. If it has an edge over the dummy model then it is providing valuable predictive insight.