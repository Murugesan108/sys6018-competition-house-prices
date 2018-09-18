install.packages("tidyverse")
install.packages("ggplot2")
install.packages("corrplot")
library(tidyverse)
library(ggplot2)
library(corrplot)

#read prediction set from csv
pred <- read_csv("test.csv")
#read training set from csv
train <- read_csv("train.csv")

###Analyze response variable first###
summary(train$SalePrice)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 34900  130000  163000  180900  214000  755000 

#We see the distribution of sale prices is not normally distributed, so
#plot the distribution to visualize better
hist1 <- ggplot(train, aes(x=SalePrice/1000))
hist1 <- hist1 + geom_histogram(binwidth=5)
hist1
#SalePrice definitely right skewed, take this into consideration for model and prediction.

###Now to analyze feature properties###
#Investigation of features determined to be most important via random forest
#Ideal number of features determined to be 9 through cross validation
#Feature Importance Ranking for top 15 features:
#   'GrLivArea',
#   'LotFrontage',
#   'LotArea',
#   'GarageArea',
#   'TotalBsmtSF',
#   'BsmtUnfSF',
#   '1stFlrSF',
#   'YearBuilt',
#   'YearRemodAdd',
#   'BsmtFinSF1',
#   'OpenPorchSF',
#   'MoSold',
#   'GarageYrBlt',
#   'WoodDeckSF',
#   'YrSold'

#***Analyze top 9 which is ideal number of features***:
#   'GrLivArea',
#   'LotFrontage',
#   'LotArea',
#   'GarageArea',
#   'TotalBsmtSF',
#   'BsmtUnfSF',
#   '1stFlrSF',
#   'YearBuilt',
#   'YearRemodAdd',

#*GrLivArea*
summary(train$GrLivArea)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 334    1130    1464    1515    1777    5642 

#Plot distribution of GrLivArea
hist2 <- ggplot(train, aes(x=GrLivArea))
hist2 <- hist2 + geom_histogram(binwidth=10)
hist2

#*LotFrontage*
summary(train$LotFrontage)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
#   21.00   59.00   69.00   70.05   80.00  313.00     259 
#Use metric of large quantity of NA's when considering imputation!

#Plot distribution of LotFrontage
hist3 <- ggplot(train, aes(x=LotFrontage))
hist3 <- hist3 + geom_histogram(binwidth=10)
hist3

#*LotArea*
summary(train$LotArea)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1300    7554    9478   10520   11600  215200 
#Seeing that the max is very high, this helps when considering outlier detection!

#Plot distribution of LotArea
hist4 <- ggplot(train, aes(x=LotArea/10000))
hist4 <- hist4 + geom_histogram(binwidth=1)
hist4

#*GarageArea*
summary(train$GarageArea)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0   334.5   480.0   473.0   576.0  1418.0 

#Plot distribution of GarageArea
hist5 <- ggplot(train, aes(x=GarageArea))
hist5 <- hist5 + geom_histogram(binwidth=10)
hist5
#Notice large number of values that are zero for this feature

#*TotalBsmtSF*
summary(train$TotalBsmtSF)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0   795.8   991.5  1057.0  1298.0  6110.0 
#Large number for max value taken into consideration with outlier detection
#Also consider the number of values at zero


#Plot distribution of TotalBsmtSF
hist6 <- ggplot(train, aes(x=TotalBsmtSF))
hist6 <- hist6 + geom_histogram(binwidth=10)
hist6

#*BsmtUnfSF*
summary(train$BsmtUnfSF)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0   223.0   477.5   567.2   808.0  2336.0 

#Plot distribution of TotalBsmtSF
hist7 <- ggplot(train, aes(x=BsmtUnfSF))
hist7 <- hist7 + geom_histogram(binwidth=10)
hist7

#*1stFlrSF*
summary(train$`1stFlrSF`)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 334     882    1087    1163    1391    4692 

#Plot distribution of 1stFlrSF
hist8 <- ggplot(train, aes(x=`1stFlrSF`))
hist8 <- hist8 + geom_histogram(binwidth=10)
hist8

#*YearBuilt*
summary(train$YearBuilt)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1872    1954    1973    1971    2000    2010 

#Plot distribution of 1stFlrSF
hist9 <- ggplot(train, aes(x=YearBuilt))
hist9 <- hist9 + geom_histogram(binwidth=10)
hist9

#*YearRemodAdd*
summary(train$YearRemodAdd)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 1950    1967    1994    1985    2004    2010 

#Plot distribution of YearRemodAdd
hist10 <- ggplot(train, aes(x=YearRemodAdd))
hist10 <- hist10 + geom_histogram(binwidth=10)
hist10

#Note that the year values follow non-right skewed distribution due to temporal setting.

#__________________________________________________________________________________
#Since we havent done imputation for categorical variables yet, 
#analyze just numeric features for heat map correlation matrix 
numeric_features <- which(sapply(train, is.numeric))
# Id    MSSubClass   LotFrontage       LotArea   OverallQual   OverallCond     YearBuilt 
# 1             2             4             5            18            19            20 
# YearRemodAdd    MasVnrArea    BsmtFinSF1    BsmtFinSF2     BsmtUnfSF   TotalBsmtSF      1stFlrSF 
# 21            27            35            37            38            39            44 
# 2ndFlrSF  LowQualFinSF     GrLivArea  BsmtFullBath  BsmtHalfBath      FullBath      HalfBath 
# 45            46            47            48            49            50            51 
# BedroomAbvGr  KitchenAbvGr  TotRmsAbvGrd    Fireplaces   GarageYrBlt    GarageCars    GarageArea 
# 52            53            55            57            60            62            63 
# WoodDeckSF   OpenPorchSF EnclosedPorch     3SsnPorch   ScreenPorch      PoolArea       MiscVal 
# 67            68            69            70            71            72            76 
# MoSold        YrSold     SalePrice 
# 77            78            81 
#Save set of numeric features
numeric_names <- names(numeric_features)
numeric_set <- train[, numeric_features]
cor_matrix <- cor(numeric_set, use="pairwise.complete.obs") 

#sort from smallest to largest correlations with SalePrice
cor_matrix2 <- as.matrix(sort(cor_matrix[,'SalePrice']))
#select only high corelations
cor_strong <- names(which(apply(cor_matrix2, 1, function(x) abs(x)>0.6)))
cor_matrix_final <- cor_matrix[cor_strong, cor_strong]
corr_plot <-corrplot.mixed(cor_matrix_final, tl.col="black", tl.pos = "lt")