import pandas as pd
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/kedjoshi/Desktop/datasets_564980_1026099_life-expectancy.csv')

######Input

def inputYears():
    '''
    This function asks the user to input yearA, yearB and entity. It checks for simple sanity and if okay, simply returns
    :return: None
    '''

    yearA = 0
    yearB = 0

    while True:

        yearA = input("Please input the starting year \n")
        yearA = int(yearA)
        yearB = input("Please input the ending year \n")
        yearB = int(yearB)

        if (yearA >= 1950 and yearA <= 2019) and (yearB >= 1950 and yearB <= 2019) and (yearA < yearB):
            break

        else:
            print("You have entered the wrong values for year. Please make sure the years are in range 1950 to 2019")

    return (yearA, yearB)

######Cleaning

def countNulls(data=dataset):

    '''
    This function calculates the null values in the dataset on every column and reports the null values in counts and percentage
    :param df: Dataset
    :return: None
    '''
    col = list(data.columns)
    totalCols = len(list(data.columns))
    colCountNulls = 0
    nullCount = 0

    print('----------------------------------------------------------------------------------------------')
    for num, column in enumerate(col):

        nullCount = data[column].isnull().sum()

        if nullCount > 0:

            colCountNulls = colCountNulls + 1

            print('{0} column \'{1}\' contains null values equal to {2}'.format(num, column, nullCount))

    print('{} columns contain null values; {} is the total number of columns'.format(colCountNulls, totalCols))
    print('----------------------------------------------------------------------------------------------')

    return

def showOutliers(data=dataset,showFigure=False):

    '''
    This function shows the datapoints as is in their raw form by plotting their boxplot and histogram
    :param data: Dataset
    :param show_plot: False by default
    :return: None
    '''

    if showFigure:
        column = data[list(data.columns)[3]]
        plt.figure(figsize=(15, 15))
        plt.subplot(2,1,1)
        plt.boxplot(column)
        plt.ylabel('Life Expectancy (Years)', fontsize=18)
        plt.title('Boxplot')
        plt.subplot(2,1,2)
        plt.hist(column)
        plt.xlabel('Life Expectancy (Years)', fontsize=18)
        plt.ylabel('Frequency', fontsize=16)
        plt.title('Histogram')
        plt.show()

def countOutliers(data=dataset):
    '''
    Counts outlier data points. The interquantile range is used as metric in setting the limits on the data. A multiple (1.5) in this case is used
    to scale the interquartile range to set the limits
    :param data: dataset
    :return: None
    '''

    col = list(data.columns)[3]

    quantile75, quantile25 = np.percentile(data[col], [75, 25])

    lowerLimit = quantile25 - ((quantile75-quantile25) *1.5)
    upperLimit = quantile75 + ((quantile75-quantile25) *1.5)

    countOutlier = len(np.where((data[col] > upperLimit) | (data[col] < lowerLimit))[0])
    percentOutlier = round(countOutlier/len(data[col])*100, 2)
    print('----------------------------------------------------------------------------------------------')
    print('Outliers: {}'.format(countOutlier))
    print('Outlier Percent: {}%'.format(percentOutlier))
    print('----------------------------------------------------------------------------------------------')

def winsorizeData(dataN, data=dataset, lowerLimit=0.0, upperLimit=0.0, showFigure=False):
    '''
    This function uses mstats.winsorize() to winsorize the data. It uses the upper and lower limits to the data.
    Any data that is above/below this limit is just nset equal to the upper and lower limits
    :param dataN: New data to be winsorized
    :param data: Current data
    :param lowerLimit: Limits on the winsorization
    :param upperLimit: Limits on the winsorization
    :param showFigure: Default to False
    :return: None
    '''

    col = list(data.columns)[3]
    dataN[col] = mstats.winsorize(data[col], limits=(lowerLimit, upperLimit))

    if showFigure:
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.boxplot(data[col])
        plt.ylabel('Life Expectancy (Years)', fontsize=18)
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(dataN[col])
        plt.ylabel('Life Expectancy (Years)', fontsize=18)
        plt.title('modified {}'.format(col))
        plt.show()

########Exploration

def entityStatistics(entity, yearA, yearB, data):

    '''
    Uses lambda function to select the rows within the yearA to yearB range together with the entity name. Standard functions
    are used to find the statistics
    :param entity: Entity for which stat is to be calculated
    :param yearA: start year
    :param yearB: end year
    :param data: Dataset
    :return: None
    '''

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB) & (data['Entity'] == entity), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following are the statistics for {0} between years {1} and {2}'.format(entity, yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(round(res.std(),3)['Life expectancy (years)']))
    print('----------------------------------------------------------------------------------------------')

    return

def globalStatistics(yearA, yearB, data):

    '''
    Uses lambda function to select the rows within the yearA to yearB range. Standard functions
    are used to find the statistics
    :param yearA: start year
    :param yearB: end year
    :param data: Dataset
    :return: None
    '''

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following are the global statistics between years {0} and {1}'.format(yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(round(res.std(),3)['Life expectancy (years)']))
    print('----------------------------------------------------------------------------------------------')

    return

def annualChangeStatistics(yearA, yearB, data):

    '''
    Uses lambda function to select the rows within the yearA to yearB range together with the entity name. The selected
    data is then iterated over with selecting consecutive year rows. That represents the annual change. Median of all such
    change values is estimated using standard functions
    :param yearA: start year
    :param yearB: end year
    :param data: dataset
    :return: None
    '''

    acData = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    res = pd.DataFrame({'Change':[]})

    for i in range(1, len(acData)):

        current_row = acData.iloc[i]
        previous_row= acData.iloc[i-1]

        if (current_row[2] == previous_row[2] + 1):

            res = res.append({'Change':current_row[3]-previous_row[3]}, ignore_index=True)

    print('----------------------------------------------------------------------------------------------')
    print('The global median life expectancy annual change from year {0} to {1} is {2}'.format(yearA, yearB, round(res.median(),2)['Change']))
    print('----------------------------------------------------------------------------------------------')

    return

def stabilityStatistics(yearA, yearB, data):

    '''
    This function first collects the rows of data that fall within the (yearA,yearB) range. We then club them together with
    'Entity' and find out the max and min for each entity. The function then picks out the row with minimum range (most stable).
    :param yearA: Start year
    :param yearB: End year
    :param data: dataset
    :return: None
    '''


    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Entity', 'Life expectancy (years)']]
    resEntity = res.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].min(), :]

    print('----------------------------------------------------------------------------------------------')
    print('The entity with most stable life expectancy between {0} and {1} is {2}'.format(yearA, yearB, resEntity.index[0]))
    print('----------------------------------------------------------------------------------------------')

    return

def percentileAnnualStatistics(yearA, yearB, data):

    '''
    This function first picks up those rows which fall in (yearA,yearB) range. It then iterates over the rows with the
    condition that we compare the consecutive years (annual change). When it is grouped by entity, the function finds out
    the rows for which lie above the 95% and finally prints them out
    :param yearA: Start year
    :param yearB: End year
    :param data: dataset
    :return:
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

    '''
    This function collects all the rows which lie in the range (yearA, yearB). Next grouping by entity, the max and min
    is found. The one entity with max range is assumed to give the highest increase in range(yearA, yearB).
    :param yearA: start year
    :param yearB: end year
    :param data: dataset
    :return:
    '''

    test = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Entity', 'Life expectancy (years)']]
    test = test.groupby(['Entity'])
    test = test.apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].max(), :]

    print('----------------------------------------------------------------------------------------------')
    print('The entity with highest increase in expectancy between {0} and {1} is {2}'.format(yearA, yearB, test.index[0]))
    print('----------------------------------------------------------------------------------------------')

    return

def calcPercent(data):

    '''
    This function simply estimates the percent increase by taking diff of max and min and dividing by min
    :param data: dataset
    :return: None
    '''

    data['Life expectancy (years)'] = (data['Life expectancy (years)'].max() - data['Life expectancy (years)'].min()) * 100 / (data['Life expectancy (years)'].min())

    return data

def quickestIncreaseStatistics(yearA, yearB, data):

    '''
    This function first picks up all the rows that lie within the limits specified (yearA, yearB). It then generates a
    unique set of those years that lie in the above range (since multiple entities might have the same year). It then
    iterates all over those unique years. For every pair of the unique year set, it picks up all the entities that have
    two data points (one at each of those paired years). Then, it calculates the percent increase for each entity, checks
    if it is greater than 40%. To find out the quickest increase, it divides the percent increase by years taken to
    achieve that change. Since we need to find the entity that made that change happen the quickest, the function sorts
    the parsed rows and picks up the top 3 entities. Thus giving the ones with a quickest change.
    :param yearA: Start year
    :param yearB: End year
    :param data: Dataset
    :return: None
    '''

    res = pd.DataFrame({'Entity1':[],'Life expectancy (years)':[]})
    data['Entity1'] = data['Entity']

    yearUnique = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Year']]['Year'].unique()

    for i in range(len(yearUnique)-1):
        for j in range(i+1, len(yearUnique)):

            test = data.loc[lambda data: (data['Year'] == yearUnique[i]) | (data['Year'] == yearUnique[j]), ['Entity','Entity1','Year','Life expectancy (years)']].groupby(['Entity'])[['Entity1','Life expectancy (years)']]
            test = test.apply(calcPercent)
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
    showOutliers(showFigure=False)
    countOutliers()

    winData = dataset.iloc[:, 0:4]

    winsorizeData(winData, dataset, lowerLimit=0.1, upperLimit=0.0,showFigure=False)

    col = list(winData.columns)

    winData[list(dataset.columns)[0]] = dataset[list(dataset.columns)[0]]
    winData[list(dataset.columns)[1]] = dataset[list(dataset.columns)[1]]
    winData[list(dataset.columns)[2]] = dataset[list(dataset.columns)[2]]
    winData['Life expectancy (years)'] = round(winData['Life expectancy (years)'],2)

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






