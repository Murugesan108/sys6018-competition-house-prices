rm(list = ls())

## Setting the working directory
path <-'./Kaggle competitions/House price prediction/data'
setwd(path)


# Calling required library
library(dplyr)

# Reading the input data

train_input <- read.csv("train.csv")
test_input <- read.csv("test.csv")

train_input$input <- "Train"
test_input$input <- "Test"

dim(train)
#[1] 1460   81
dim(test)
test_input$SalePrice <- NA
#[1] 1459   80

train_all <- rbind(train_input,test_input)

########### DATA ANALYSIS FOR PRE-PROCESSING ####################################

#Finding unique values under each column
unique_counts_df <- data.frame(unique_values = sapply(train,function(x)length(unique(x))))
unique_counts_df$col_names <- row.names(unique_counts_df)
categories <- unique_counts_df[unique_counts_df$unique_values <= 2,]

#Finding the number of NAs across each column
NA_counts_df <- data.frame(NA_values = sapply(train,function(x)sum(is.na(x))))
NA_counts_df$columns <- row.names(NA_counts_df)

#Top NA values
Top_NA_values <- NA_counts_df %>% arrange(desc(NA_values)) %>% 
                mutate(perc = round(NA_values/nrow(train) * 100,2)) %>% filter(NA_values != 0)
#Top 5 features cannot be used even for NA interpolation since we have high number of NA values - above ~50%
# We can interpolate the values from 'GarageType' to 'Electrical'


############### NA TREATMENT #########################


train_all$PoolArea <- NULL
train_all$MiscFeature <- NULL
train_all$Alley <- NULL
train_all$Fence  <- NULL
train_all$FireplaceQu <- NULL


train_all$GarageType[is.na(train_all$GarageType)] <- "Attchd"




train_all[is.na(train_all)] <- 0



######################################################################


# Building a baseline linear model
remove_cols <- c(Top_NA_values[1:6,"columns"],"Exterior1st","ExterCond","Foundation","Heating")

lm_formula <- formula(paste0("SalePrice ~ . "))

lm_model <- lm(SalePrice ~ . , data = train[!names(train) %in% remove_cols])
#summary(lm_model)

test_preds <- predict(lm_model,test)

test_preds[is.na(test_preds)] <- mean(test_preds,na.rm= TRUE)


submission <- data.frame(Id = test$Id, SalePrice = test_preds)
#write.csv(submission,"Test_submission.csv",row.names = FALSE)

########### EXPLORATORY DATA ANALYSIS #################


############### FEATURE ENGINEERING ################




### Selecting the final set of columns for the output 
remove_cols <- names(train_all)[!sapply(train_all,is.integer)]
req_cols <- names(train_all)[!names(train_all) %in% c(remove_cols,"Id")]


train_all_scale <- train_all
train_all_scale[req_cols] <- data.frame(apply(train_all[req_cols],2,scale))
train_all_scale$SalePrice <- train_all$SalePrice
 

############## TRAINING AND HOLDOUT SET ##############


train <- train_all_scale[train_all_scale$input == "Train",]
test <-  train_all_scale[train_all_scale$input == "Test",]

set.seed(0)
train_records <- sample(1:nrow(train),0.8 * nrow(train))
holdout_records <- (1:nrow(train))[!(1:nrow(train)) %in% train_records]

train_data <- train[train_records,]
holdout_data <- train[holdout_records,]

######### PARAMETRIC MODEL APPROACH ###############


######## NON PARAMETRIC MODEL APPROACH ############
### KNN ###

#Finding the best parameters using holdout set

k <- 25
req_cols <- req_cols[!req_cols %in% c("SalePrice")]
holdout_price_all <- c()

for(records in 1:nrow(holdout_data)){
  
  ## Converting the training data to matrix
  df1 <- train_data[,req_cols]
  row.names(df1) <- NULL
  df1 <- as.matrix(df1)
  
  ## Converting the holdout data to matrix
  df2 <-  holdout_data[records,req_cols]
  row.names(df2) <- NULL
  df2 <- as.matrix(df2)
  
  distance_each <- sqrt(rowSums((as.vector(df2) - df1)^2))
  names(distance_each) <- 1:length(distance_each)
  
  positions <- head(sort(distance_each),k)
  
  weights <- 1 - positions/sum(positions)
  
  all_price <- train_data[as.numeric(names(positions)),"SalePrice"]
  
  holdout_price <- sum(all_price * weights)/sum(weights)
  holdout_price_all <- c(holdout_price_all,holdout_price)
  
  
}

sum((holdout_price_all - holdout_data$SalePrice)^2)/1e10




#### Final prediction ### 


k <- 40
req_cols <- req_cols[!req_cols %in% "SalePrice"]
test_price_all <- c()

for(records in 1:nrow(test)){
  
  ## Converting the training data to matrix
  df1 <- train[,req_cols]
  row.names(df1) <- NULL
  df1 <- as.matrix(df1)
  
  ## Converting the holdout data to matrix
  df2 <-  test[records,req_cols]
  row.names(df2) <- NULL
  df2 <- as.matrix(df2)
  
  distance_each <- sqrt(rowSums((as.vector(df2) - df1)^2))
  names(distance_each) <- 1:length(distance_each)
  
  positions <- head(sort(distance_each),k)
  
  weights <- 1 - positions/sum(positions)
  
  all_price <- train[as.numeric(names(positions)),"SalePrice"]
  
  test_price <- sum(all_price * weights,na.rm = TRUE)/sum(weights,na.rm = TRUE)
  test_price_all <- c(test_price_all,test_price)
  if(is.na(test_price)){
    stop(records)
  }
  
  
}


length(test_price_all)

submission <- data.frame(Id = test_input$Id, SalePrice = test_price_all)
write.csv(submission,"Test_submission.csv",row.names = FALSE)



