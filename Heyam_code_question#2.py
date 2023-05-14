import pandas as pd
import missingno as msno
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import math

data = pd.read_csv('Life Expectancy Data.csv')

data.info()
print(data.head())

#looking null values

print(data.isnull().sum())

#Splitting the dataset to categorical and numerical for later use
categoricalFeats = data.select_dtypes(include=[np.object])
print(categoricalFeats.columns)

numericalFeats = data.select_dtypes(include=[np.number])
print(numericalFeats.columns)

print(data.corr()['Life expectancy '])

plt.scatter(data['Life expectancy '],data['Income composition of resources'])
plt.xlabel('Life expectancy ')
plt.ylabel('Income composition of resources')
plt.show()

plt.scatter(data['Life expectancy '],data['Population'])
plt.xlabel('Life expectancy ')
plt.ylabel('Population')
plt.show()

plt.scatter(data['Life expectancy '],data['Measles '])
plt.xlabel('Life expectancy ')
plt.ylabel('Measles ')
plt.show()

plt.scatter(data['Life expectancy '],data['GDP'])
plt.xlabel('Life expectancy ')
plt.ylabel('GDP')
plt.show()

numericalFeatsMissingValues = numericalFeats.isnull().sum().sort_values(ascending=False)
numericalFeatsMissingValuesPerc = ((numericalFeats.isnull().sum()/numericalFeats.isnull().count())*100).sort_values(ascending=False)
numericalFeatsMissingData = pd.concat([numericalFeatsMissingValues, numericalFeatsMissingValuesPerc], axis=1,join='outer', keys=['Total Missing Numerical Values', '% of Total Observations'])
numericalFeatsMissingData.index.name =' Numerical Feature'
print(numericalFeatsMissingData)

categoricalFeatsMissingValues = categoricalFeats.isnull().sum().sort_values(ascending=False)
categoricalFeatsMissingValuesPerc = (categoricalFeats.isnull().sum()/categoricalFeats.isnull().count()).sort_values(ascending=False)
categoricalFeatsMissingData = pd.concat([categoricalFeatsMissingValues, categoricalFeatsMissingValuesPerc], axis=1,join='outer', keys=['Total Missing Categorical Values', '% of Total Observations'])
categoricalFeatsMissingData.index.name =' Categorical Feature'
print(categoricalFeatsMissingData)

print(data.columns)

country_list = data.Country.unique()
fill_list = ['Life expectancy ','Adult Mortality','Alcohol','Hepatitis B',' BMI ','Polio','Total expenditure','Diphtheria ','GDP','Population',' thinness  1-19 years',' thinness 5-9 years','Income composition of resources','Schooling']

for country in country_list:
    data.loc[data['Country'] == country,fill_list] = data.loc[data['Country'] == country,fill_list].interpolate()
    
# Drop remaining null values after interpolation.
data.dropna(inplace=True)

data.drop('Status', inplace=True, axis=1)
data.drop('Country', inplace=True, axis=1)
# data.drop('GDP', inplace=True, axis=1)
# data.drop('Hepatitis B', inplace=True, axis=1)
# data.drop('Total expenditure', inplace=True, axis=1)
# data.drop('Income composition of resources', inplace=True, axis=1)
# data.drop('Schooling', inplace=True, axis=1)
# data.drop('Alcohol', inplace=True, axis=1)
# data.dropna(axis = 0, how = 'any', inplace = True)

# Life_expectancy= data['Life expectancy '].mean()
# data['Life expectancy '].fillna(Life_expectancy,inplace=True)

# Adult_Mortality= data['Adult Mortality'].mean()
# data['Adult Mortality'].fillna(Adult_Mortality,inplace=True)

# Alcohol= data['Alcohol'].mean()
# data['Alcohol'].fillna(Alcohol,inplace=True)

# Hepatitis_B= data['Hepatitis B'].mean()
# data['Hepatitis B'].fillna(Hepatitis_B,inplace=True)

# BMI= data[' BMI '].mean()
# data[' BMI '].fillna(BMI,inplace=True)

# Polio= data['Polio'].mean()
# data['Polio'].fillna(Polio,inplace=True)

# Total_expenditure = data['Total expenditure'].mean()
# data['Total expenditure'].fillna(Total_expenditure,inplace=True)

# Diphtheria = data['Diphtheria '].mean()
# data['Diphtheria '].fillna(Diphtheria,inplace=True)

# GDP = data['GDP'].mean()
# data['GDP'].fillna(GDP,inplace=True)

# Population = data['Population'].mean()
# data['Population'].fillna(Population,inplace=True)

# thinness_1_19_years = data[' thinness  1-19 years'].mean()
# data[' thinness  1-19 years'].fillna(thinness_1_19_years,inplace=True)

# thinness_5_9_years = data[' thinness 5-9 years'].mean()
# data[' thinness 5-9 years'].fillna(thinness_5_9_years,inplace=True)

# Income_composition_of_resources = data['Income composition of resources'].mean()
# data['Income composition of resources'].fillna(Income_composition_of_resources,inplace=True)

# Schooling = data['Schooling'].mean()
# data['Schooling'].fillna(Schooling,inplace=True)

print(data.isnull().sum())
    

# data1 = ce.OneHotEncoder(data['Country'])
# data2=  ce.OneHotEncoder(data['Status'])   
# data4 = data1.fit_transform(data)
# data=data2.fit_transform(data)

data = data/np.linalg.norm(data)

target = data.iloc[:,3].values
Features = data.drop('Life expectancy ', axis = 1)

print("The shape of the independent fatures are ",Features.shape)
print("The shape of the dependent fatures are ",target.shape)

n_train = math.floor(0.8 * Features.shape[0])
n_test = math.ceil((1-0.8) * Features.shape[0])

X_train = Features[:n_train]
Y_train = target[:n_train]
X_test = Features[n_train:]
Y_test = target[n_train:]
print(X_train.shape)


X_train_column_name=pd.DataFrame(X_train)
X_train_column_name=X_train_column_name.columns.values
print("the names of column in X train are: ",X_train_column_name)

X_train = np.vstack((np.ones((X_train.shape[0], )),X_train.T)).T
X_test = np.vstack((np.ones((X_test.shape[0], )),X_test.T)).T

Y_train= Y_train.reshape(X_train.shape[0],1)
Y_test= Y_test.reshape(X_test.shape[0],1)

print("The shape of X_train:", X_train.shape)
print("The shape of Y_train:", Y_train.shape)
print("The shape of X_test:", X_test.shape)
print("The shape of Y_test:", Y_test.shape)




print("-----------------------------------------------------------")
def model(X, Y, iteration):
    m = Y.size
    theta = np.zeros((X.shape[1], 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/m)*np.sum(np.square(y_pred-Y))
        d_theta = (2/m)*np.dot(X.T, (y_pred -Y ))
        theta = theta - 0.1*d_theta
        cost_list.append(cost)
        print("cost is",cost)
    return theta, cost_list



iteration = 100
theta, cost_list = model(X_train,Y_train,iteration)

theta_abs=np.abs(theta)
j=1

for i in range(0, len(X_train_column_name)):
    print("the importance of feature",X_train_column_name[i],": " ,theta_abs[j])
    j+=1

print("--------------------------------------------------------------")

theta_sort=sorted(theta_abs,reverse=True)

for i in range (1,len(theta_abs)):
    for j in range (1,len(theta_abs)):
        if (theta_abs[j] == theta_sort[i]):
            print("the most important feature is:",X_train_column_name[j-1],"with ", theta_abs[j])
  


rng=np.arange(0,iteration)
plt.plot(rng ,cost_list)
plt.show()

prediction =np.dot(X_test,theta)
error = (2/X_test.shape[0])*np.sum(np.square(prediction-Y_test))
print("Test error is:",error*100,"%")
print("Test Accuracy is:",(1-error)*100,"%")
