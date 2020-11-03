import pandas as pd
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/kedjoshi/Desktop/datasets_564980_1026099_life-expectancy.csv')

######Cleaning

def countNulls(df=dataset):
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

def showOutliers(data=dataset):

    column = data[list(data.columns)[3]]
    plt.figure(figsize=(15, 40))
    plt.subplot(2,1,1)
    plt.boxplot(column)
    plt.title('Boxplot')
    plt.subplot(2,1,2)
    plt.hist(column)
    plt.title('Histogram')
    #plt.show()

def countOutliers(data=dataset):

    col = list(data.columns)[3]
    print(15*'-' + col + 15*'-')
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    print('Number of outliers: {}'.format(outlier_count))
    print('Percent of data that is outlier: {}%'.format(outlier_percent))

def winsorizeData(dataN, data=dataset, lower_limit=0.0, upper_limit=0.0, show_plot=False):

    col = list(data.columns)[3]
    dataN[col] = mstats.winsorize(data[col], limits=(lower_limit, upper_limit))
    if show_plot == True:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.boxplot(data[col])
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(dataN[col])
        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))
        #plt.show()

########Exploration



def entityStatistics(entity, yearA, yearB, data):

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB) & (data['Entity'] == entity), :]

    print('The following are the statistics for {0} between years {1} and {2}'.format(entity, yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(res.std()['Life expectancy (years)']))

    return

def globalStatistics(yearA, yearB, data):

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    print('The following are the global statistics between years {0} and {1}'.format(yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(res.std()['Life expectancy (years)']))

    return

def annualChangeStatistics(yearA, yearB, data):

    acData = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    res = pd.DataFrame({'Change':[]})

    for i in range(1, len(acData)):

        current_row = acData.iloc[i]
        previous_row= acData.iloc[i-1]

        if (current_row[2] == previous_row[2] + 1):

            res = res.append({'Change':current_row[3]-previous_row[3]}, ignore_index=True)

    print('The global median life expectancy annual change from year {0} to {1} is {2}'.format(yearA, yearB, res.median()['Change']))

    return

'''
#Q4

test = data.loc[lambda data: (data['Year'] > 1989) & (data['Year'] < 2020), ['Entity', 'Life expectancy (years)']]

print(test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].min(), :])

#Q5

test = data
test = test.loc[lambda test: (test['Year'] == 1969) | (test['Year'] == 2020), ['Entity', 'Life expectancy (years)']]
test = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] > x['Life expectancy (years)'].quantile(0.95), :]

print(test)

#Q6

test = data
test = test.loc[lambda test: (test['Year'] > 1969) & (test['Year'] < 2020), ['Entity', 'Life expectancy (years)']]
test = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].max(), :]

print(test)

#Q3-II

test = data.loc[lambda data: (data['Year'] > 1969) & (data['Year'] < 2020), :]

df = pd.DataFrame({'Entity':[], 'change':[]})

for i in range(1, len(test)):

    current_row = test.iloc[i]
    previous_row =test.iloc[i-1]

    if (current_row[2] == previous_row[2] + 1):

        df = df.append({'Entity': current_row[0],'change':current_row[3]-previous_row[3]}, ignore_index=True)

df = df.groupby(['Entity']).apply(lambda x: x.max()).loc[lambda x: x['change'] > x['change'].quantile(0.95), :]

print(df)


#Q7
def calc(data):

    data['Life expectancy (years)'] = (data['Life expectancy (years)'].max() - data['Life expectancy (years)'].min()) * 100 / (data['Life expectancy (years)'].min()*(j-i))

    return data

res = pd.DataFrame({'Entity1':[],'Life expectancy (years)':[]})
data['Entity1'] = data['Entity']

for i in range(1950, 1980):
    for j in range(i+1, 1980):

        test = data.loc[lambda data: (data['Year'] == i) | (data['Year'] == j), ['Entity','Entity1','Life expectancy (years)']]

        res1 = test.groupby(['Entity'])[['Entity1','Life expectancy (years)']]
        res1 = res1.apply(calc)
        res1 = pd.DataFrame(res1)
        res1 = res1.loc[lambda x: x['Life expectancy (years)'] *(j-i) > 40, :]

        res = res.append(res1)

print(res.sort_values(by=['Life expectancy (years)']).groupby(['Entity1']).describe())

'''

if __name__ == "__main__":

    expCol = list(dataset.columns)[3]

    plt.boxplot(dataset[expCol])
    #plt.show()
    plt.hist(dataset[expCol])
    #plt.show()

    countNulls()
    showOutliers()
    countOutliers()

    winData = dataset.iloc[:, 0:4]

    winsorizeData(winData, dataset, lower_limit=0.1, upper_limit=0.0, show_plot=True)

    col = list(winData.columns)

    winData[list(dataset.columns)[0]] = dataset[list(dataset.columns)[0]]
    winData[list(dataset.columns)[1]] = dataset[list(dataset.columns)[1]]
    winData[list(dataset.columns)[2]] = dataset[list(dataset.columns)[2]]

    entityStatistics('Austria', 1980, 2005, winData)
    globalStatistics(1980, 2005, winData)
    annualChangeStatistics(1980, 2005, winData)





