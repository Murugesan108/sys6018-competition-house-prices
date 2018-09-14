rm(list = ls())

## Setting the working directory
path <-'./Kaggle competitions/House price prediction/data'
setwd(path)


# Calling required library
library(dplyr)

# Reading the input data

train <- read.csv("train.csv")
test <- read.csv("test.csv")

dim(train)
#[1] 1460   81

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
Top_NA_values


#Top 5 features cannot be used even for NA interpolation since we have high number of NA values - above ~50%
# We can interpolate the values from 'GarageType' to 'Electrical'

#Analysing the columns with top NA values
View(train[Top_NA_values$columns[5:length(Top_NA_values$columns)]])


# Building a baseline linear model

remove_cols <- c(Top_NA_values[1:6,"columns"],"Exterior1st","ExterCond","Foundation","Heating")

lm_formula <- formula(paste0("SalePrice ~ . "))

lm_model <- lm(SalePrice ~ . , data = train[!names(train) %in% remove_cols])
#summary(lm_model)

test_preds <- predict(lm_model,test)

test_preds[is.na(test_preds)] <- mean(test_preds,na.rm= TRUE)


submission <- data.frame(Id = test$Id, SalePrice = test_preds)
write.csv(submission,"Test_submission.csv",row.names = FALSE)





########### EXPLORATORY DATA ANALYSIS #################

########### NA TREATMENT #########################

########## FEATURE ENGINEERING ################


######### PARAMETRIC MODEL APPROACH ###############


######## NON PARAMETRIC MODEL APPROACH ############




### KNN ###

remove_cols <- names(train)[!sapply(train,is.integer)]

data = train[!names(train) %in% c(remove_cols,remove_cols,"SalePrice","Id")]
data_test = test[!names(test) %in% c(remove_cols,remove_cols,"Id")]



head(data)
dim(data)
dim(data_test)


data[is.na(data)] <- 0
data_test[is.na(data_test)] <- 0

data_all <- rbind(data,data_test)


data_all_scale <- data.frame(apply(data_all,2,scale))

train_matrix <- as.matrix(data_all_scale[1:nrow(data),])
test_matrix <- (as.matrix(data_all_scale[ (nrow(data)+1):nrow(data_all_scale), ]))

k <- 3
test_val <- c()

for(each in 1:nrow(test_matrix)){
  
  #print(each)
  distance_with_each_training <- sapply(rowSums((test_matrix[each,] - train_matrix)^2),sqrt)
  top_position <- head(sort(distance_with_each_training),k)
  top_position_names <- names(head(sort(distance_with_each_training),k))
  
  weights <- sum(top_position) - top_position
  test_val <- c(test_val,sum(weights * train[as.numeric(top_position),]$SalePrice)/sum(weights))

}

submission <- data.frame(Id = test$Id, SalePrice = test_val)
write.csv(submission,"Test_submission.csv",row.names = FALSE)



