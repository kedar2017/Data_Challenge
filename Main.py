import pandas as pd
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/kedjoshi/Desktop/datasets_564980_1026099_life-expectancy.csv')

#Cleaning
'''
column = dataset[list(dataset.columns)[3]]

def nulls_breakdown(df=dataset):
    df_cols = list(df.columns)
    cols_total_count = len(list(df.columns))
    cols_count = 0
    for loc, col in enumerate(df_cols):
        null_count = df[col].isnull().sum()
        total_count = df[col].isnull().count()
        percent_null = round(null_count/total_count*100, 2)
        if null_count > 0:
            cols_count += 1
            print('[iloc = {}] {} has {} null values: {}% null'.format(loc, col, null_count, percent_null))
        cols_percent_null = round(cols_count/cols_total_count*100, 2)
    print('Out of {} total columns, {} contain null values; {}% columns contain null values.'.format(cols_total_count, cols_count, cols_percent_null))

def outliers_visual(data=dataset):
    plt.figure(figsize=(15, 40))
    plt.subplot(2,1,1)
    plt.boxplot(column)
    plt.title('Boxplot')
    plt.subplot(2,1,2)
    plt.hist(column)
    plt.title('Histogram')
    #plt.show()

def outlier_count(col, data=dataset):
    print(15*'-' + col + 15*'-')
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    print('Number of outliers: {}'.format(outlier_count))
    print('Percent of data that is outlier: {}%'.format(outlier_percent))

def test_wins(col, lower_limit=0.0, upper_limit=0.0, show_plot=True):
    wins_data = mstats.winsorize(dataset[col], limits=(lower_limit, upper_limit))
    wins_dict[col] = wins_data
    if show_plot == True:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.boxplot(dataset[col])
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(wins_data)
        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))
        #plt.show()


plt.boxplot(dataset[list(dataset.columns)[3]])
plt.show()
plt.hist(dataset[list(dataset.columns)[3]])
plt.show()

nulls_breakdown()
outliers_visual()
outlier_count(list(dataset.columns)[3])
'''

def test_wins(col, lower_limit=0.0, upper_limit=0.0, show_plot=True):
    wins_data = mstats.winsorize(dataset[col], limits=(lower_limit, upper_limit))
    wins_dict[col] = wins_data
    if show_plot == True:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.boxplot(dataset[col])
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(wins_data)
        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))
        #plt.show()

wins_dict = {}
wins_df = dataset.iloc[:, 0:4]

test_wins(list(dataset.columns)[3], lower_limit=0.1, upper_limit=0.0, show_plot=True)

col = list(wins_df.columns)

wins_df[list(dataset.columns)[0]] = dataset[list(dataset.columns)[0]]
wins_df[list(dataset.columns)[1]] = dataset[list(dataset.columns)[1]]
wins_df[list(dataset.columns)[2]] = dataset[list(dataset.columns)[2]]
wins_df[list(dataset.columns)[3]] = wins_dict[list(dataset.columns)[3]]
#print(wins_df.describe())

#Exploration

'''
#Q1
print(wins_df[col[3]].median())

#Q2
print(wins_df.loc[lambda wins_df: (wins_df['Year'] > 1969) & (wins_df['Year'] < 2020), :].median())

#Q3
test = wins_df.loc[lambda wins_df: (wins_df['Year'] > 1969) & (wins_df['Year'] < 2020), :]

d = {'change': []}
df = pd.DataFrame(d)

for i in range(1, len(test)):

    current_row = test.iloc[i]
    previous_row =test.iloc[i-1]

    if (current_row[2] == previous_row[2] + 1):

        df = df.append({'change':current_row[3]-previous_row[3]}, ignore_index=True)

print(df.describe())

#Q4

test = wins_df.loc[lambda wins_df: (wins_df['Year'] > 1989) & (wins_df['Year'] < 2020), ['Entity', 'Life expectancy (years)']]

print(test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].min(), :])

#Q5

test = wins_df
test = test.loc[lambda test: (test['Year'] == 1969) | (test['Year'] == 2020), ['Entity', 'Life expectancy (years)']]
test = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] > x['Life expectancy (years)'].quantile(0.95), :]

print(test)

#Q6

test = wins_df
test = test.loc[lambda test: (test['Year'] > 1969) & (test['Year'] < 2020), ['Entity', 'Life expectancy (years)']]
test = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].max(), :]

print(test)

#Q3-II

test = wins_df.loc[lambda wins_df: (wins_df['Year'] > 1969) & (wins_df['Year'] < 2020), :]

df = pd.DataFrame({'Entity':[], 'change':[]})

for i in range(1, len(test)):

    current_row = test.iloc[i]
    previous_row =test.iloc[i-1]

    if (current_row[2] == previous_row[2] + 1):

        df = df.append({'Entity': current_row[0],'change':current_row[3]-previous_row[3]}, ignore_index=True)

df = df.groupby(['Entity']).apply(lambda x: x.max()).loc[lambda x: x['change'] > x['change'].quantile(0.95), :]

print(df)

'''

#Q7

test = wins_df.loc[lambda wins_df: (wins_df['Year'] > 1955) & (wins_df['Year'] < 1975), ['Entity', 'Life expectancy (years)']]
test = test.groupby(['Entity']).apply(lambda x: (x.max()-x.min())*100/x.min()).loc[lambda x: x['Life expectancy (years)'] > 40, :]

print(test)
