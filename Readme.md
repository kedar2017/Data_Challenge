# Data Challenge - Life Expectancy

The documentation is written as follows.

The first section discusses the packages needed for running the modules. Next, an example usage is shown. Following which the data cleaning is discussed and finally the data exploration.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install pandas
pip install numpy
pip install scipy
pip install matplotlib
```
Pandas, numpy, scipy.stats and matplotlib.pyplot are used in the module.

## Usage

You will be presented with the following prompts to input the year range and entity values. The code checks the sanity
of the entered year values.

```python

#################################################################################################
Hello we will ask you to input a few values to get the module started
Please input the starting year
1990
Please input the ending year
2000
Please input the entity
'Austria'
#################################################################################################
```

## Data Cleaning

1. Variables
2. Missing values
3. Outliers

Variables are studied by looking at what each of the variables signify and what are there data types. The following is used for the same:

```python
dataset = pd.read_csv('/datasets_564980_1026099_life-expectancy.csv')

dataset #Print statement

########################OUTPUT#########################
            Entity Code  Year  Life expectancy (years)
0      Afghanistan  AFG  1950                   27.638
1      Afghanistan  AFG  1951                   27.878
2      Afghanistan  AFG  1952                   28.361
3      Afghanistan  AFG  1953                   28.852
4      Afghanistan  AFG  1954                   29.350
...            ...  ...   ...                      ...
19023     Zimbabwe  ZWE  2015                   59.534
19024     Zimbabwe  ZWE  2016                   60.294
19025     Zimbabwe  ZWE  2017                   60.812
19026     Zimbabwe  ZWE  2018                   61.195
19027     Zimbabwe  ZWE  2019                   61.490
#######################################################

dataset['Code']

########################OUTPUT#########################
0        AFG
1        AFG
2        AFG
3        AFG
4        AFG
        ...
19023    ZWE
19024    ZWE
19025    ZWE
19026    ZWE
19027    ZWE
Name: Code, Length: 19028, dtype: object
#######################################################

dataset['Entity']

########################OUTPUT#########################
0        Afghanistan
1        Afghanistan
2        Afghanistan
3        Afghanistan
4        Afghanistan
            ...
19023       Zimbabwe
19024       Zimbabwe
19025       Zimbabwe
19026       Zimbabwe
19027       Zimbabwe
Name: Entity, Length: 19028, dtype: object
#######################################################

dataset['Year']

########################OUTPUT#########################
0        1950
1        1951
2        1952
3        1953
4        1954
         ...
19023    2015
19024    2016
19025    2017
19026    2018
19027    2019
Name: Year, Length: 19028, dtype: int64
#######################################################

dataset['Life expectancy (years)']

########################OUTPUT#########################
0        27.638
1        27.878
2        28.361
3        28.852
4        29.350
          ...
19023    59.534
19024    60.294
19025    60.812
19026    61.195
19027    61.490
Name: Life expectancy (years), Length: 19028, dtype: float64
#######################################################

dataset.describe()

########################OUTPUT#########################
               Year  Life expectancy (years)
count  19028.000000             19028.000000
mean    1974.955171                61.751767
std       38.157409                13.091632
min     1543.000000                17.760000
25%     1961.000000                52.314750
50%     1980.000000                64.713000
75%     2000.000000                71.984250
max     2019.000000                86.751000
#######################################################

dataset.info()

########################OUTPUT#########################
RangeIndex: 19028 entries, 0 to 19027
Data columns (total 4 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   Entity                   19028 non-null  object
 1   Code                     18445 non-null  object
 2   Year                     19028 non-null  int64
 3   Life expectancy (years)  19028 non-null  float64
dtypes: float64(1), int64(1), object(2)

#######################################################


```

Shows that the 'Year' and 'Life Expectancy' values are int and float types and equal in number in terms of total counts.

Next, we look at the missing values. We see from the above info() that there are 19028 values for each 'Entity', 'Year' and 'Life expectancy' but only 18445 values for 'Code'. Which means there are quite a few 'Code' rows with nulls. Upon looking up closely it seems that the missing 'Code' values correspond to the continents and rest of the codes with existing values are for countries. This doesn't have a great impact on our calculations since we are looking up everything with respect to 'Entities' and hence both continents and countries are considered as separate entities. The number of such null values are found out using the following function. As shown, there is only one column ('Code') with Null values

```python
def countNulls(df=dataset):
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

#Here is the sample output from above

Code has 583 null values: 3.06% null
Out of 4 total columns, 1 contain null values; 25.0% columns contain null values.
```

Now that we looked at missing values, we can try to figure out if there are any outliers in our data. Let's look closely at the above describe() function.

```python

dataset.describe()

########################OUTPUT#########################
               Year  Life expectancy (years)
count  19028.000000             19028.000000
mean    1974.955171                61.751767
std       38.157409                13.091632
min     1543.000000                17.760000
25%     1961.000000                52.314750
50%     1980.000000                64.713000
75%     2000.000000                71.984250
max     2019.000000                86.751000
#######################################################

```




The mean, min and max values for the 'Year' seem to be fine. We have to consider the minimum value '1543' for the year as it is one of the extreme points.

But, looking at the 'Life Expectancy' there seems to be something worth discussing. The minimum life expectancy number is close to 17. Which seems a bit odd. So we look at filtering the data to limit the range of life expectancy.

We first look at the boxplot and histogram to see if we can see any outliers as pointed out above. The following figures show how the life expectancy data is distributed.

The below function shows the distribution:

```python

def showOutliers(data=dataset):

    column = data[list(data.columns)[3]]
    plt.figure(figsize=(15, 40))
    plt.subplot(2,1,1)
    plt.boxplot(column)
    plt.title('Boxplot')
    plt.subplot(2,1,2)
    plt.hist(column)
    plt.title('Histogram')
    plt.show()

```

![alt text](https://github.com/kedar2017/Data_Challenge/blob/main/Figure_2.png)


We consider the metric 'Interquartile Range' to count how many values lie outside the range and mark them as 'outliers'. We consider some multiple of the IQR as a measure for counting the outliers.

multiple = 1.5 for the below function

```python

def countOutliers(data=dataset):

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
```

In order to deal with the outliers, we follow a technique called 'Winsorization' to limit the upper and lower bounds. We check the limits as per the above formula and whichever values lie outside this range are simply made equal to the lower and upper limits using the mstats.winsorize() function.

```python

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
        plt.show()
```

Below is the plot showing the boxplot comparing the winsorized data and original data.

![alt text](https://github.com/kedar2017/Data_Challenge/blob/main/Figure_1.png)


So the dataN is our new data that we will use for all the next functions.

## Data Exploration

Statistics on the following are discussed in detail in this section:

1. Entity based statistics
2. Global statistics
3. Annual change
4. Stability
5. Annual change percentile
6. Increased expectancy
7. Quickest increase

Pandas library is used throughout this exercise.

Starting with entity based stats, the following function is written.

```python

def entityStatistics(entity, yearA, yearB, data):

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB) & (data['Entity'] == entity), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following are the statistics for {0} between years {1} and {2}'.format(entity, yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(res.std()['Life expectancy (years)']))
    print('----------------------------------------------------------------------------------------------')

```

The above function first selects all the data that is within the given year range (yearA, yearB). It then compares the
'entity' column with the value provided for which we want to estimate the statistics.

The res object is then queried with max, min, median etc. to give out the results for that particular entity.


Next, to estimate the global statistics, we just select the rows with 'Year' column values within the range provided.
The statistics are then printed out by querying the res object.

```python

def globalStatistics(yearA, yearB, data):

    res = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), :]

    print('----------------------------------------------------------------------------------------------')
    print('The following are the global statistics between years {0} and {1}'.format(yearA, yearB))
    print('Median: {}'.format(res.median()['Life expectancy (years)']))
    print('Maximum: {}'.format(res.max()['Life expectancy (years)']))
    print('Minimum: {}'.format(res.min()['Life expectancy (years)']))
    print('Standard Dev.: {}'.format(res.std()['Life expectancy (years)']))
    print('----------------------------------------------------------------------------------------------')

    return

```

Moving to the stability statistics, we again look at the rows with values for 'Year' columns that lie in the range
provided.

We then estimate the stability of the life expectancy increase by looking at the maximum and minimum values within
that year range for each entity. The entity with the minimum range (maximum - minimum) value is selected to be the one having highest
stability. See the funtion below:

```python

def stabilityStatistics(yearA, yearB, data):


    data['Entity1'] = data['Entity']
    test = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Entity', 'Life expectancy (years)']]
    resEntity = test.groupby(['Entity']).apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].min(), :]

    print('----------------------------------------------------------------------------------------------')
    print('The entity with most stable life expectancy between {0} and {1} is {2}'.format(yearA, yearB, resEntity.index[0]))
    print('----------------------------------------------------------------------------------------------')

    return

```

The next step is to discuss the statistics related to increase in expectancy over a range of years. Same logic applies
here like the earlier function. We look at the maximum and minimum values within the year range provided. Then apply
the maximum of the difference/increase (max - min). The entity with that highest increase is then selected.

```python

def highestIncreaseStatistics(yearA, yearB, data):

    test = data.loc[lambda data: (data['Year'] >= yearA) & (data['Year'] <= yearB), ['Entity', 'Life expectancy (years)']]
    test = test.groupby(['Entity'])
    test = test.apply(lambda x: x.max()-x.min()).loc[lambda x: x['Life expectancy (years)'] == x['Life expectancy (years)'].max(), :]

    print('----------------------------------------------------------------------------------------------')
    print('The entity with highest increase in expectancy between {0} and {1} is {2}'.format(yearA, yearB, test.index[0]))
    print('----------------------------------------------------------------------------------------------')

    return
```

Moving on to the annual change statistics, the logic used here is that we start with the provided year range.
For each of the year selected in that range, we estimate the change in life expectancy over the one year period.
All the annual life expectancy changes for each of the rows (within the year range provided) are recorded. The median
is then estimated for the above recorded rows.

```python

def annualChangeStatistics(yearA, yearB, data):

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

```

As shown above, the acData stores the rows which have 'Year' within the range of years provided (yearA, yearB).
A res data frame is created that holds the 'Change' column. Going row by row, the change is calculated only for rows
where the delta year is equal to 1 (annual change). This eliminates the cases in which there are different entities on
adjacent rows. The assumption is that all the 'year' values for each entity are arranged in increasing order.

Going through the res rows that holds the annual changes, we simply call the median() function to estimate the
'global annual change' statistics.

Next, we look at the percentile annual change statistic. The approach is exactly similar to the above function.
For each of the year selected in that range, we estimate the change in life expectancy over the one year period.
All the annual life expectancy changes for each of the rows (within the year range provided) are recorded. The median
is then estimated for the above recorded rows. oing row by row, the change is calculated only for rows
where the delta year is equal to 1 (annual change). This eliminates the cases in which there are different entities on
adjacent rows.

Next, we look at the maximum annual change in expectancy (stored in row annualD) grouped by 'Entity'. So we are now
left with every row having an entity together with their own maximum annual change over the course of given range of years.

Next, among the entities, we find the 95% quantile number. Whichever entities exceed that quantile number are then selected
and printed out.

```python

def percentileAnnualStatistics(yearA, yearB, data):

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

```

Moving onto the most time consuming computation, finding the entity with quickest increase in the life expectancy
by over 40%.

The logic used for this is as follows. A res dataframe is first generated.

We go through a range of year numbers. An outer loop with 'i' runs between yearA and yearB. An inner loop with pointer 'j',
runs from 'i+1' to yearB. For a particular (i,j) pair, all the rows with year i and year j are selected. The life
expectancy increase is estimated going from year i to year j for eacj of the entity. The percent increase in the
expectancy is calculated for this particular (i,j) pair as:

% increase = (expectancy[j] - expectancy[i]) / (expectancy[i])

Note above that we have used the fact that expectancy[j] = max() and expectancy[i] = min() for a particular entity.
Assumption is that there is always an increase in the expectancy with respect to year. Percent increase is hence estimated.
Next, we also have to estimate the 'quickest' increase. Hence, the entities for which the percent increase is greater than
40% is collected together. The obtained number is then divided by (j-i) [the difference in year]. So, we ideally want to
find out which entity has the highest value for (% increase)/ (j-i).

The above process is iterated for different values of i and j that run over an outer and inner loop on the year range.

Finally, we sort the values obtained from the above process. Note that, there will be multiple values for a single entity
obtained because multiple (i,j) will get a percent increase > 40% for a single entity. So, after sorting the values, the
top 3 unique entities are selected. Just to make it easier to code, an extra utility column ('Entity1') has been added
so that the entity values are not lost after the groupby() function is used. The rest of the coding steps, speak for themselves.



```python

def quickestIncreaseStatistics(yearA, yearB, data):

    res = pd.DataFrame({'Entity1':[],'Life expectancy (years)':[]})
    data['Entity1'] = data['Entity']

    for i in range(yearA, yearB+1):
        for j in range(i+1, yearB+1):

            test = data.loc[lambda data: (data['Year'] == i) | (data['Year'] == j), ['Entity','Entity1','Life expectancy (years)']]

            res1 = test.groupby(['Entity'])[['Entity1','Life expectancy (years)']]
            res1 = res1.apply(calc)
            res1 = pd.DataFrame(res1)
            res1 = res1.loc[lambda x: x['Life expectancy (years)'] > 40, :]
            res1['Life expectancy (years)'] = res1['Life expectancy (years)'] / (j-i)
            res = res.append(res1)

    res = res.sort_values(by=['Life expectancy (years)'], ascending=False)['Entity1'].unique()[:3]

    print('----------------------------------------------------------------------------------------------')
    print('The entities with quickest increase in expectancy by 40% are as follows ')

    for i in res:
        print(i)
    print('----------------------------------------------------------------------------------------------')

    return
```

