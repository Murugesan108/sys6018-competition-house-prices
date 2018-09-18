# sys6018-competition-house-prices
Kaggle competition #2 to predict the house prices

Predicting house prices using two methods

1) Parametric (Linear Regression,...)

Linear Regression is easy to interpret
2) Non-parametric approach (knn method)

Given a point x, calulate the distance of point x between all its neighours. And get K nearest neighbours. The prediction is weighted average, which means nearer neighbour would be more important in prediciton. The equation is (1-dist/total_dist)* SalePrice. We use cross validation to determine the K.

Project Members:
1) Andrew Dahbura
2) Boda Ye
3) Murugesan Ramakrishnan

To start the project, Andrew worked on exploratory data analysis, visualization and data exploration
in order to better understand the data. Data exploration involved investigating the response variable,
including its summary and distribution, and then investigating the relationship between the response
variable and various features. This includes correlation matrix production, boxplots, and further visualization.

After data exploration, Murugesan worked on data cleaning which involves missing value imputation, 
outlier detection and assesment, and other cleaning processes. NA values for categorical variables will
be imputed using the mode, and imputation for other features will involve clustering or class divisions to 
help segment the data before using mean value imputation.

Boda will use input from EDA and data cleaning to begin feature selection process. This entails feature 
importance ranking, determining ideal number of features, etc.

After all of this has been completed, Andrew, Murugesan and Boda will work together to build the optimal
parametric (Linear regression) and nonparametric (KNN) model for analysis.
