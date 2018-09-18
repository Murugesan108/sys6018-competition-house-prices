### Competition 2: House Prices
## Team C1-7
## Members: Boda Ye, Andrew Dahbura, Murugesan Ramakrishnan
#########################################################################################

# Reading in the libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#import numpy as np

#Setting the working directory
path = "C:\\Users\\arvra\\Documents\\UVa files\\Classes\\Fall_18\\Data Mining\\Kaggle competitions\\House price prediction\\data"
os.chdir(path)

#Reading in the train and test datasets 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Converting test and test into a single file 'combined file.csv'
file= pd.concat([train,test], axis = 0)

############# EXPLORATORY DATA ANLYSIS CARRIED OUT IN R #############################

############### DATA ANALYSIS FOR PRE-PROCESSING ####################################

#Checking the number of NA values for each column
NA_values = file.isna().sum()

#Checking the % of NA values for each column
Top_NA_values = 100 * NA_values.sort_values(ascending=False)/file.shape[0]

Top_NA_values[Top_NA_values > 40]


#Based on the numbers we will remove the Features with NA values over ~50%

cols_to_drop = ['PoolArea','MiscFeature','Alley','Fence','FireplaceQu']

file.drop(cols_to_drop, axis = 1,inplace = True)
file.shape

unimportant_cols = [ 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',  'BsmtFinType2', 'BsmtQual', 'Condition2',
                     'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'LotConfig', 'PoolQC',
                    'RoofMatl', 'Street', 'Utilities']
file.drop(unimportant_cols, axis = 1,inplace = True)

####################### Outlier Treatment ##############################
#Looping through each of the numerical variables and capping the values with the maximum value(3 times the SD)
for each in range(file.shape[1]):
    if(file.iloc[:,each].dtype == "int64"):
        file.iloc[:,each][np.abs(file.iloc[:,each] - file.iloc[:,each].mean()) >= 3 * file.iloc[:,each].std()] = (3 * file.iloc[:,each].std())

############## IMPUTING NA VALUES #############################
#### Categorical Column
   
#Selecting certain top Categorical columns and replacing the missing values with their corresponding model values
file.MasVnrType[file.MasVnrType.isna()] = "None"
file.Electrical[file.Electrical.isna()] = "SBrkr"

#file.GarageType[file.GarageType.isna()] = "Attchd"
#file.GarageFinish[file.GarageFinish.isna()] = "Unf"
#file.GarageQual[file.GarageQual.isna()] = "TA"
#file.BsmtFinType1[file.BsmtFinType1.isna()] = "Unf"
#file.GarageCond[file.GarageCond.isna()] = "TA"
  
  #Looping through other categorical and replacing them with the mode values
  
for each in Top_NA_values.index:
  if(each in file.columns and file.loc[:,each].dtype not in ["float64","int64"]):
     file.loc[:,each][file.loc[:,each].isna()] = file.loc[:,each].value_counts()[0:1].index
      
   
##Selecting the categorical variables and creating a one-hot vector for each of them
x=file.iloc[:,:-1]
columns=['MSZoning',	'LotShape',	'LandContour',	'LandSlope','Neighborhood',	'Condition1',	'BldgType',	'HouseStyle',\
        'RoofStyle','Exterior1st','Exterior2nd','MasVnrType',	'ExterQual',	'ExterCond',	'Foundation',\
        'Heating',	'HeatingQC',	'CentralAir',	'Electrical',	'KitchenQual',	'Functional',\
        'PavedDrive',	'SaleType',	'SaleCondition']
x_dummy=pd.get_dummies(x,columns=columns,drop_first=True)
x_dummy

#Get train variables
x_train=x_dummy.iloc[:1460,1:].values
x_train

#Impute the NA values with the mean values
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='mean',axis=0)
imputer.fit(x_train[:,:])
x_train[:,:]=imputer.transform(x_train[:,:])


######################## Preparing the data for modeling #############################

#Getting the SalePrice (reponse variable) values
y_train=file['SalePrice'][:1460].values
y_train


##### RANDOM FOREST MODEL - to get the top features to be used for modeling
#random feature importance
forest = RandomForestClassifier()
forest.fit(x_train,y_train)
importances = forest.feature_importances_

# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
newColumnOrder=[]
for k,v in sorted(zip(map(lambda x: round(x, 4), importances), x_dummy.columns[1:]), reverse=True):
    print(str(v)+':'+str(k))
    newColumnOrder.append(v)
newColumnOrder

#reorder the columns according to their importance
x_dummy_reind = x_dummy.reindex(newColumnOrder, axis=1)
x_dummy_reind=x_dummy_reind.values
imputer.fit(x_dummy_reind[:,:])
x_dummy_reind[:,:]=imputer.transform(x_dummy_reind[:,:])
#scale features
from sklearn.preprocessing import scale
x_dummy_reind=scale(x_dummy_reind)
x_dummy_reind
x_train_reind=x_dummy_reind[:1460]
len(x_train_reind[0])

######### USING CROSS VALIDATION TO SELECT THE TOP N FEATURES (FINDING OUT N) ######################################
#this part is used to find the best features
#cross validation function
from sklearn.model_selection import KFold
# This program does 5-fold. It saves the result at each time as different parts of y_pred.
# In the end, it returns the y_pred as the result of all the five 5-fold.
def run_cv(X, y, clf_class, **kwargs):
    # X is the denpendent variables, y is the response variable
    # Construct a kfolds object
    kf = KFold(n_splits=5, shuffle=True)
    y_pred = y.copy()
    clf = clf_class(**kwargs)
    # Iterate through folds
    # train_index, and test_index should be lists which holds randomly chosen data
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]

        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    # get a whole pred from the orignal data
    return y_pred



#This part is used to decide the number of top K fatures we're going to use in modeling
from sklearn.metrics import mean_squared_error
import heapq
locHeap=[]
#use cross validation to determine how many features are we going to remain
for i in range(5):
    for num_feats in range(5,15):
        #get the top number of features
        x_train_feats=x_train_reind[:,:num_feats]

        #do cross validation
        RF_CV_result = run_cv(x_train_feats, y_train, RandomForestClassifier)
        #print("Random forest: " +str(num_feats)+ '  '+str(mean_squared_error(y_train, RF_CV_result)))
        heapq.heappush(locHeap,(mean_squared_error(y_train, RF_CV_result),num_feats))


#this part is used to top K features we're using
check=[]
for i in range(20):
    check.append(heapq.heappop(locHeap)[1])
check

#get variable variables
x_pred=x_dummy_reind[1460:,1:10]
x_train_reind=x_train_reind[:,1:10]
len(x_train_reind)
len(x_pred)

x_pred


############################ PARAMETRIC APPROACH #####################################

### Linear Regression
## Since the response varaible is continuous and we might need to see the significance for each features
## we move ahead with performing linear regression
#Linear regression is the best to get significance of features and it is easy to explain

from sklearn.linear_model import LinearRegression,Lasso
Linear_result = run_cv(x_train_reind, y_train, LinearRegression)
mean_squared_error(y_train, Linear_result)


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
# Train the model using the training sets
regr.fit(x_train_reind, y_train)
linear_pred=regr.predict(x_pred)
prediction_linear=pd.DataFrame({'Id':range(1461,2920),'SalePrice':linear_pred})
prediction_linear.to_csv('prediction_linear.csv', index = False)


############################## NON PARAMETRIC APPROACH #############################
############################# IMPLEMENTATION OF KNN ###################################

#input x train, y train, and x_pred,and
import heapq
from scipy.spatial.distance import euclidean
def KNN(x_train,y_train,x_pred,k):
    y_pred=[]
    for pred in x_pred:
        minHeap=[]
        for x in range(1168):
            dist = euclidean(pred,x_train[x])
            heapq.heappush(minHeap,(dist,x))

        y_bucket=[]
        for i in range(k):
            y_bucket.append(heapq.heappop(minHeap))

        dist_total=sum(item[0] for item in y_bucket)

        y_pred.append(sum(y_train[a[1]]*a[0]/dist_total for a in y_bucket))

    return y_pred



#cross validation to get the best K
# Run time : about a couple of minutes    
for i in range(3,7):
    kf_cv=KFold(n_splits=5)
    kf_cv.get_n_splits(x_train_reind)
    y_pred_cv=y_train.copy()
    for train_index, test_index in kf_cv.split(x_train_reind):
        x_train_cv, x_test_cv = x_train_reind[train_index], x_train_reind[test_index]
        y_train_cv = y_train[train_index]
        y_pred_cv[test_index]=KNN(x_train_cv,y_train_cv,x_test_cv,i)

    print(str(i)+' '+str(mean_squared_error(y_train,y_pred_cv)))

y_pred=KNN(x_train_reind,y_train,x_pred,5)
prediction=pd.DataFrame({'Id':range(1461,2920),'SalePrice':y_pred})
prediction.to_csv('prediction_knn.csv',index = False)
prediction
