#######################################
#this part is used to preprocess the data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
combined=pd.concat([train,test])
combined.to_csv('combined.csv')






file=pd.read_csv('combined file.csv')
#get y_train
y_train=file['SalePrice'][:1460].values
y_train

file.shape


#dummy the independent variables
x=file.iloc[:,:-1]
columns=['MSZoning',	'LotShape',	'LandContour',	'LandSlope','Neighborhood',	'Condition1',	'BldgType',	'HouseStyle',\
        'RoofStyle','Exterior1st','Exterior2nd','MasVnrType',	'ExterQual',	'ExterCond',	'Foundation',\
        'Heating',	'HeatingQC',	'CentralAir',	'Electrical',	'KitchenQual',	'Functional',\
        'PavedDrive',	'SaleType',	'SaleCondition'
]
x_dummy=pd.get_dummies(x,columns=columns,drop_first=True)
x_dummy


#get train variables
x_train=x_dummy.iloc[:1460,1:].values
x_train

#delete NAs
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='mean',axis=0)
imputer.fit(x_train[:,:])
x_train[:,:]=imputer.transform(x_train[:,:])


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


#########################################################
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







#############################
#KNN
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

        y_pred.append(sum(y_train[a[1]]*(1-a[0]/dist_total) for a in y_bucket))

    return y_pred


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
prediction_linear.to_csv('prediction_linear.csv')











#cross validation to get the best K
for i in range(3,7):
    kf_cv=KFold(n_splits=5)
    kf_cv.get_n_splits(x_train_reind)
    y_pred_cv=y_train.copy()
    for train_index, test_index in kf_cv.split(x_train_reind):
        x_train_cv, x_test_cv = x_train_reind[train_index], x_train_reind[test_index]
        y_train_cv = y_train[train_index]
        y_pred_cv[test_index]=KNN(x_train_cv,y_train_cv,x_test_cv,i)

    print(str(i)+' '+str(mean_squared_error(y_train,y_pred_cv)))




y_pred=KNN(x_train_reind,y_train,x_pred,3)
prediction=pd.DataFrame({'Id':range(1461,2920),'SalePrice':y_pred})
prediction.to_csv('prediction.csv')
prediction


####################################
#heap map
corr = x[["MSSubClass","LotFrontage","OverallQual","OverallCond","YearBuilt","LotArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",]].corr()
corr


import matplotlib.pyplot as plt
import seaborn as sb
sb.heatmap(corr)
plt.savefig('myfig.png')






