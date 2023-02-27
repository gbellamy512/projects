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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# set seed for reproducibility
random_seed = 1

#
## Download the data
#
df = pd.read_csv('ewma_df_2016_2022.csv')
# Calculate the home team margin of victory (negative is a loss) which will be used as the regression target.
df['home_margin'] = df['home_score'] - df['away_score']
# Drop columns with >=5% null
# https://stackoverflow.com/questions/43311555/how-to-drop-column-according-to-nan-percentage-for-dataframe
df = df.loc[:, df.isnull().mean() < .05]

#
## Exploratory data analysis
#
print(df.head())
df.info()
# Make sure all 32 teams are present and equally represented
print('team count:', len(df["home_team"].value_counts()))
print(df["home_team"].value_counts())
# All 32 teams are present but some teams have more occurrences than others. This is due to more playoff appearances.
# Let's filter out weeks later than 17 and see if it better balances the distribution across teams.
df = df[df["week"] <= 17]
print(df["home_team"].value_counts())
# Looks better!

# Let's take a look at the distributions of our numerical variables by plotting their histograms.
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
df.hist(bins=50, figsize=(12, 8))
# plt.show()

# Let's set the target and check how variables correlate with it
target = 'home_margin'
corr_matrix = df.corr()
print(corr_matrix[target].sort_values(ascending=False))
# Unsurprisingly, passing_offense_home is most correlated with the home team win margin (although still not a strong
# correlation at only ~.2). It’s not surprising that rushing defense has the lowest correlation with winning margin
# but it is that nearly no correlation exists (<.01).
# Let's plot some of the passing correlations to see them visually
attributes = ["home_margin", "ewma_dynamic_window_passing_offense_home", "ewma_dynamic_window_passing_defense_home",
              "ewma_dynamic_window_passing_offense_away", "ewma_dynamic_window_passing_defense_away"]
scatter_matrix(df[attributes], figsize=(12, 8))
# plt.show()
df.plot(kind="scatter", x="ewma_dynamic_window_passing_offense_home", y="home_margin", alpha=0.1, grid=True)
# plt.show()

#
## Feature Engineering
#
# let's create a summer winter variable b/c scoring may decrease in the winter when snow is more likely and to be
# honest because we want to have a categorical variable to deal with
# https://stackoverflow.com/questions/60285557/extract-seasons-from-datetime-pandas
df['game_date_home']= pd.to_datetime(df['game_date_home'])
df['date_offset'] = (df.game_date_home.dt.month*100 + df.game_date_home.dt.day - 320)%1300
df['season_weather'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                      labels=['spring', 'summer', 'autumn', 'winter'])
# above creates dtype 'category'
df['season_weather'] = df['season_weather'].astype(str)

#
## Create a test set
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

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
param_distribs = {
    # default = 100
    'random_forest__n_estimators': randint(low=25, high=1_000)
    # default = 1
    , 'random_forest__max_features': randint(low=2, high=100)
    # default = None
    , 'random_forest__max_depth': randint(low=2, high=100)
    # default = 2
    , 'random_forest__min_samples_split': randint(low=2, high=100)
    # default = 1
    , 'random_forest__min_samples_leaf': randint(low=1, high=100)
}
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=random_seed)
# https://stackoverflow.com/questions/58313842/a-column-vector-y-was-passed-when-a-1d-array-was-expected-error-message
rnd_search.fit(X_train, np.ravel(y_train))
y_pred = rnd_search.predict(X_test)
print('rs mse: {mse}'.format(mse=mean_squared_error(y_test, y_pred)))
print('rs root of mse: {mse}'.format(mse=math.sqrt(mean_squared_error(y_test, y_pred))))

final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
importance_tuples = sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
           reverse=True)
for (score, name) in importance_tuples:
    print(round(score, 2), name)

# https://stackoverflow.com/questions/40729162/merging-results-from-model-predict-with-original-pandas-dataframe
X_test['y'] = y_test
X_test['pred'] = y_pred
X_test['error'] = X_test['pred'] - X_test['y']
X_test.to_csv('rfr.csv')

# ~13.5 error is really bad. Let's see if Elastic Net Regression works better.
from sklearn.linear_model import ElasticNet

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

# again almost ~13.5 error. Doesn't seem like these inputs  can produce a quality regression for margin of victory.

# Let's see if classification will work better.

df = pd.read_csv('ewma_df_2016_2022.csv')
df = df.loc[:, df.isnull().mean() < .05]
df = df[df["week"] <= 17]
drop_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'game_date_home'
    , 'game_date_away', 'month', 'day', 'year']
df.drop(drop_cols, axis=1, inplace=True)
target = 'home_team_win'
y = pd.DataFrame(df, columns=[target])
X = df.drop([target], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=random_seed)),
        ('rf', RandomForestClassifier(random_state=random_seed)),
        ('svc', SVC(random_state=random_seed))
    ]
)
voting_clf.fit(X_train, y_train)

for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test,y_test))

print('hard voting score:', voting_clf.score(X_test, y_test))
# soft voting picks the estimator with the highest probability as opposed to hard voting which averages the set
voting_clf.voting = 'soft'
voting_clf.named_estimators['svc'].probability = True
voting_clf.fit(X_train, y_train)
print('soft voting score:', voting_clf.score(X_test, y_test))
# n_iter_no_change implements early stopping
gbclf_best = GradientBoostingClassifier(max_depth=2, learning_rate=0.05, n_estimators=500, n_iter_no_change=10
                                        , random_state=42)
gbclf_best.fit(X_train, y_train)
print('early stopping number of estimators:', gbclf_best.n_estimators_)
print('early stopping:', gbclf_best.score(X_test, y_test))

# hard voting was the best of the bunch, achieving a 63% score.