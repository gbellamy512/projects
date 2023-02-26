import pandas as pd

df = pd.read_csv('epa_schedule_df_2016_2022_2.csv')
df['home_margin'] = df['home_score'] - df['away_score']

# Looking for Correlations

corr_matrix = df.corr()

print(corr_matrix["home_margin"].sort_values(ascending=False))