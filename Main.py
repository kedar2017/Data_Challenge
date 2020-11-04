import pandas as pd
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/kedjoshi/Desktop/datasets_564980_1026099_life-expectancy.csv')

######Input

def inputYears():
    '''
    This function asks the user to input yearA, yearB and entity. It checks for simple sanity and if okay, simply returns
    :return:
    '''

    yearA = 0
    yearB = 0

    while True:

        yearA = input("Please input the starting year \n")
        yearA = int(yearA)
        yearB = input("Please input the ending year \n")
        yearB = int(yearB)

        if (yearA >= 1550 and yearA <= 2019) and (yearB >= 1550 and yearB <= 2019) and (yearA < yearB):
            break

        else:
            print("You have entered the wrong values for year")

    return (yearA, yearB)


######Cleaning

def countNulls(df=dataset):

    '''
    This function calculates the null values in the dataset on every column and reports the null values in counts and percentage
    :param df: Dataset
    :return:
    '''
    df_cols = list(df.columns)
    cols_total_count = len(list(df.columns))
    cols_count = 0

    print('----------------------------------------------------------------------------------------------')
    for loc, col in enumerate(df_cols):
        null_count = df[col].isnull().sum()
        total_count = df[col].isnull().count()
        percent_null = round(null_count/total_count*100, 2)

        if null_count > 0:
            cols_count += 1
            print('{} column has {} null values: {}% null'.format(loc, col, null_count, percent_null))
        cols_percent_null = round(cols_count/cols_total_count*100, 2)
    print('Out of {} total columns, {} contain null values; {}% columns contain null values.'.format(cols_total_count, cols_count, cols_percent_null))
    print('----------------------------------------------------------------------------------------------')

    return

def showOutliers(data=dataset,show_plot=False):

    '''
    This function shows the datapoints as is in their raw form by plotting their boxplot and histogram
    :param data: Dataset
    :param show_plot: False by default
    :return:
    '''

    if show_plot == True:
        column = data[list(data.columns)[3]]
        plt.figure(figsize=(15, 40))
        plt.subplot(2,1,1)
        plt.boxplot(column)
        plt.title('Boxplot')
        plt.subplot(2,1,2)
        plt.hist(column)
        plt.title('Histogram')
        plt.show()

def countOutliers(data=dataset):
    '''
    Counts outlier data points. The interquantile range is used as metric in setting the limits on the data. A multiple (1.5) in this case is used
    to scale the interquartile range to set the limits
    :param data: dataset
    :return:
    '''

    col = list(data.columns)[3]
    print(15*'-' + col + 15*'-')
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    print('----------------------------------------------------------------------------------------------')
    print('Number of outliers: {}'.format(outlier_count))
    print('Percent of data that is outlier: {}%'.format(outlier_percent))
    print('----------------------------------------------------------------------------------------------')

def winsorizeData(dataN, data=dataset, lower_limit=0.0, upper_limit=0.0, show_plot=False):
    '''
    This function uses mstats.winsorize() to winsorize the data. It uses the upper and lower limits to the data.
    Any data that is above/below this limit is just nset equal to the upper and lower limits
    :param dataN:
    :param data:
    :param lower_limit:
    :param upper_limit:
    :param show_plot:
    :return:
    '''

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
        plt.show()

########Exploration

def entityStatistics(entity, yearA, yearB, data):

    '''
    Uses lambda function to select the rows within the yearA to yearB range together with the entity name. Standard functions
    are used to find the statistics
    :param entity:
    :param yearA: start year
    :param yearB: end year
    :param data:
    :return:
    '''

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB) & (data['Entity'] == entity), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following are the statistics for {0} between years {1} and {2}'.format(entity, yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(res.std()['Life expectancy (years)']))
    print('----------------------------------------------------------------------------------------------')

    return

def globalStatistics(yearA, yearB, data):

    '''
    Uses lambda function to select the rows within the yearA to yearB range. Standard functions
    are used to find the statistics
    :param entity:
    :param yearA: start year
    :param yearB: end year
    :param data:
    :return:
    '''

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following are the global statistics between years {0} and {1}'.format(yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(res.std()['Life expectancy (years)']))
    print('----------------------------------------------------------------------------------------------')

    return

def annualChangeStatistics(yearA, yearB, data):

    '''
    Uses lambda function to select the rows within the yearA to yearB range together with the entity name. The selected
    data is then iterated over with selecting consecutive year rows. That represents the annual change. Median of all such
    change values is estimated using standard functions
    :param entity:
    :param yearA: start year
    :param yearB: end year
    :param data:
    :return:
    '''

    acData = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    res = pd.DataFrame({'Change':[]})

    for i in range(1, len(acData)):

        current_row = acData.iloc[i]
        previous_row= acData.iloc[i-1]

        if (current_row[2] == previous_row[2] + 1):

            res = res.append({'Change':current_row[3]-previous_row[3]}, ignore_index=True)

    print('----------------------------------------------------------------------------------------------')
    print('The global median life expectancy annual change from year {0} to {1} is {2}'.format(yearA, yearB, res.median()['Change']))
    print('----------------------------------------------------------------------------------------------')

    return

def stabilityStatistics(yearA, yearB, data):


    data['Entity1'] = data['Entity']
    test = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Entity', 'Life expectancy (years)']]
    resEntity = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].min(), :]

    print('----------------------------------------------------------------------------------------------')
    print('The entity with most stable life expectancy between {0} and {1} is {2}'.format(yearA, yearB, resEntity.index[0]))
    print('----------------------------------------------------------------------------------------------')

    return

def percentileAnnualStatistics(yearA, yearB, data):

    '''
    test = data.loc[lambda data: (data['Year'] == yearA) | (data['Year'] == yearB), ['Entity', 'Life expectancy (years)']]
    test = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] > x['Life expectancy (years)'].quantile(0.95), :]
    '''

    test = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    annualD = pd.DataFrame({'Entity': [], 'change': []})

    for i in range(1, len(test)):

        current_row = test.iloc[i]
        previous_row = test.iloc[i - 1]

        if current_row[2] == previous_row[2] + 1:
            annualD = annualD.append({'Entity': current_row[0], 'change': current_row[3] - previous_row[3]}, ignore_index=True)

    annualD = annualD.groupby(['Entity']).apply(lambda x: x.max()).loc[lambda x: x['change'] > x['change'].quantile(0.95), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following entities reported above 95th percentile highest annual life expectancy increase between {0} and {1}: '.format(yearA, yearB))

    for i in annualD.index:
        print(i)
    print('----------------------------------------------------------------------------------------------')

    return

def highestIncreaseStatistics(yearA, yearB, data):

    test = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Entity', 'Life expectancy (years)']]
    test = test.groupby(['Entity'])
    test = test.apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].max(), :]

    print('----------------------------------------------------------------------------------------------')
    print('The entity with highest increase in expectancy between {0} and {1} is {2}'.format(yearA, yearB, test.index[0]))
    print('----------------------------------------------------------------------------------------------')

    return

def calc(data):

    data['Life expectancy (years)'] = (data['Life expectancy (years)'].max() - data['Life expectancy (years)'].min()) * 100 / (data['Life expectancy (years)'].min())

    return data

def quickestIncreaseStatistics(yearA, yearB, data):

    res = pd.DataFrame({'Entity1':[],'Life expectancy (years)':[]})
    data['Entity1'] = data['Entity']

    yearUnique = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Year']]['Year'].unique()

    for i in range(len(yearUnique)-1):
        for j in range(i+1, len(yearUnique)):

            test = data.loc[lambda data: (data['Year'] == yearUnique[i]) | (data['Year'] == yearUnique[j]), ['Entity','Entity1','Year','Life expectancy (years)']].groupby(['Entity'])[['Entity1','Life expectancy (years)']]
            test = test.apply(calc)
            test = pd.DataFrame(test)
            test = test.loc[lambda x: x['Life expectancy (years)'] > 40, :]
            test['Life expectancy (years)'] = test['Life expectancy (years)'] / (yearUnique[j]-yearUnique[i])
            res = res.append(test)

    res = res.sort_values(by=['Life expectancy (years)'], ascending=False)['Entity1'].unique()[:3]

    print('----------------------------------------------------------------------------------------------')
    print('The entities with quickest increase in expectancy by 40% are as follows ')
    for i in res:
        print(i)
    print('----------------------------------------------------------------------------------------------')

    return

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

    winsorizeData(winData, dataset, lower_limit=0.1, upper_limit=0.0)

    col = list(winData.columns)

    winData[list(dataset.columns)[0]] = dataset[list(dataset.columns)[0]]
    winData[list(dataset.columns)[1]] = dataset[list(dataset.columns)[1]]
    winData[list(dataset.columns)[2]] = dataset[list(dataset.columns)[2]]

    print("Hello we will ask you to input a few values to get the module started")

    (yearA, yearB) = inputYears()

    entity= input("Please input the entity \n")
    
    entityStatistics(entity, yearA, yearB, winData)
    globalStatistics(yearA, yearB, winData)
    annualChangeStatistics(yearA, yearB, winData)
    stabilityStatistics(yearA, yearB, winData)
    percentileAnnualStatistics(yearA, yearB, winData)
    highestIncreaseStatistics(yearA, yearB, winData)
    quickestIncreaseStatistics(yearA, yearB, winData)






