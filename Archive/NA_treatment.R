na_treatment <- function(train_all,Top_NA_values){
  
  ### Removin the columns with large number of NA values
  train_all$PoolArea <- NULL
  train_all$MiscFeature <- NULL
  train_all$Alley <- NULL
  train_all$Fence  <- NULL
  train_all$FireplaceQu <- NULL
  
  
  ### Categorical Columns
  ##Replacing the NA with the model values
  train_all$GarageType[is.na(train_all$GarageType)] <- "Attchd"
  train_all$GarageFinish[is.na(train_all$GarageFinish)] <- "Unf"
  train_all$GarageQual[is.na(train_all$GarageQual)] <- "TA"
  train_all$GarageCond[is.na(train_all$GarageCond)] <- "TA"
  train_all$BsmtFinType1[is.na(train_all$BsmtFinType1)] <- "Unf"
  train_all$MasVnrType[is.na(train_all$MasVnrType)] <- "None"
  train_all$Electrical[is.na(train_all$Electrical)] <- "SBrkr"
  
  ##Replacing Categorical NA values with the mode of each column
  for(each in Top_NA_values$columns){
    train_all[each][is.na(train_all[each])] <- names(which.max(table(train_all[each])))
    
  }
  
  
  ##Names of the numerical columns with NA values
  NA_cols <- sapply(train_all, function(x)any(is.na(x)))
  
  ### Numerical Columns
  for(each in names(train_all)[NA_cols]){
    train_all[each][is.na(train_all[each])] <- mean(train_all[each],na.rm = TRUE)
  }
  
  
  
  
  train_all[is.na(train_all)] <- 0
  
  
  return(train_all)
}