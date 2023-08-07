# Libraries
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(tidyr)
library(reshape2)
library(ggpubr)
library(tidyverse)
library(randomForest)
library(ggplot2)
library(xgboost)
library(e1071)
library(rpart)


# Filling nulls with mode function
Mode <- function(x) {
  sort(-table(x))[1]
}


# Reading data
data <- read.csv("D:\Uni\8th semester\Distributed Computing\project\\train.csv")
test_data <- read.csv("D:\Uni\8th semester\Distributed Computing\project\\test.csv")

#saving ID column
ID <- test_data$Id

############################## Pre processing on TRAIN data ##############################


#Filling Nulls

data$X1stFlrSF[is.na(data$X1stFlrSF)] <- mean(data$X1stFlrSF,na.rm = TRUE)
data$X2ndFlrSF[is.na(data$X2ndFlrSF)] <- mean(data$X2ndFlrSF,na.rm = TRUE)
data$TotalBsmtSF[is.na(data$TotalBsmtSF)] <- mean(data$TotalBsmtSF,na.rm = TRUE)
data$GarageArea[is.na(data$GarageArea)] <- mean(data$GarageArea,na.rm = TRUE)
data$GrLivArea[is.na(data$GrLivArea)] <- mean(data$GrLivArea,na.rm = TRUE)
data$PoolArea[is.na(data$PoolArea)] <- mean(data$PoolArea,na.rm = TRUE)

data$YearRemodAdd[is.na(data$YearRemodAdd)] <- Mode(data$YearRemodAdd)
data$YearBuilt[is.na(data$YearBuilt)] <- Mode(data$YearBuilt)
data$BsmtFullBath[is.na(data$BsmtFullBath)] <- Mode(data$BsmtFullBath)
data$BsmtHalfBath[is.na(data$BsmtHalfBath)] <- Mode(data$BsmtHalfBath)
data$FullBath[is.na(data$FullBath)] <- Mode(data$FullBath)
data$HalfBath[is.na(data$HalfBath)] <- Mode(data$HalfBath)


# Encoding columns used in data engineering then filling nulls 

data$GarageCond <- as.numeric(factor(data$GarageCond, levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5))) 
data$GarageQual  <- as.numeric(factor(data$GarageQual , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))

data$GarageCond[is.na(data$GarageCond)] <- 0
data$GarageQual[is.na(data$GarageQual)] <- 0


#Data Engineering
data$Garage <- data$GarageCond + data$GarageQual
data$totalSF<- data$TotalBsmtSF + data$X1stFlrSF + data$X2ndFlrSF 
data$builtAndRemodled <- data$YearBuilt + data$YearRemodAdd
data$totalBathrooms <- data$FullBath*2 + data$HalfBath + data$BsmtFullBath*2 + data$BsmtHalfBath


# Encoding columns with levels

data$ExterQual <- as.numeric(factor(data$ExterQual, levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5))) 
data$HeatingQC  <- as.numeric(factor(data$HeatingQC , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))
data$KitchenQual   <- as.numeric(factor(data$KitchenQual  , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))
data$FireplaceQu   <- as.numeric(factor(data$FireplaceQu , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))


#General Encoding

data$Foundation   <- as.numeric(factor(data$Foundation))
data$CentralAir   <- as.numeric(factor(data$CentralAir))


# Filling nulls after Encoding and Data Engineering

data$FireplaceQu[is.na(data$FireplaceQu)] <- 0
data$ExterQual[is.na(data$ExterQual)] <- mean(data$ExterQual,na.rm = TRUE)
data$HeatingQC[is.na(data$HeatingQC)] <- 0
data$KitchenQual[is.na(data$KitchenQual)] <- 0
data$totalSF[is.na(data$totalSF)] <- mean(data$totalSF,na.rm = TRUE)
data$GarageCars[is.na(data$GarageCars)] <- 0
data$GarageYrBlt[is.na(data$GarageYrBlt)] <- mean(data$GarageYrBlt,na.rm = TRUE)


# scatter plot between features
#plot( data$SalePrice,data$GarageCars,  xlab = "SalePrice", ylab = "GarageCars", main = "Scatter Plot between SalePrice and GarageCars")

# Choosing columns according to the highest correlations with saleprice
data <- subset(data, select = c('OverallQual', 'builtAndRemodled','totalBathrooms', 'GrLivArea','GarageArea','ExterQual', 'HeatingQC', 
                                'KitchenQual','TotRmsAbvGrd','Fireplaces','FireplaceQu','totalSF','Foundation','CentralAir','Garage','GarageCars','GarageYrBlt','SalePrice'))


head(data)









############################## Pre processing on TEST data ##############################


# Checking nulls

sum(is.na(test_data$Fence))
sum(is.na(test_data$FireplaceQu))
sum(is.na(test_data$WoodDeckSF))

#Filling Nulls

test_data$X1stFlrSF[is.na(test_data$X1stFlrSF)] <- mean(test_data$X1stFlrSF,na.rm = TRUE)
test_data$X2ndFlrSF[is.na(test_data$X2ndFlrSF)] <- mean(test_data$X2ndFlrSF,na.rm = TRUE)
test_data$TotalBsmtSF[is.na(test_data$TotalBsmtSF)] <- mean(test_data$TotalBsmtSF,na.rm = TRUE)
test_data$GarageArea[is.na(test_data$GarageArea)] <- mean(test_data$GarageArea,na.rm = TRUE)
test_data$GrLivArea[is.na(test_data$GrLivArea)] <- mean(test_data$GrLivArea,na.rm = TRUE)

test_data$YearRemodAdd[is.na(test_data$YearRemodAdd)] <- Mode(test_data$YearRemodAdd)
test_data$YearBuilt[is.na(test_data$YearBuilt)] <- Mode(test_data$YearBuilt)
test_data$BsmtFullBath[is.na(test_data$BsmtFullBath)] <- Mode(test_data$BsmtFullBath)
test_data$BsmtHalfBath[is.na(test_data$BsmtHalfBath)] <- Mode(test_data$BsmtHalfBath)
test_data$FullBath[is.na(test_data$FullBath)] <- Mode(test_data$FullBath)
test_data$HalfBath[is.na(test_data$HalfBath)] <- Mode(test_data$HalfBath)


# Encoding columns used in data engineering then filling nulls 

test_data$GarageCond <- as.numeric(factor(test_data$GarageCond, levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5))) 
test_data$GarageQual  <- as.numeric(factor(test_data$GarageQual , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))

test_data$GarageCond[is.na(test_data$GarageCond)] <- 0
test_data$GarageQual[is.na(test_data$GarageQual)] <- 0


# Encoding columns with levels

test_data$ExterQual <- as.numeric(factor(test_data$ExterQual, levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5))) 
test_data$HeatingQC  <- as.numeric(factor(test_data$HeatingQC , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))
test_data$KitchenQual   <- as.numeric(factor(test_data$KitchenQual  , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))
test_data$FireplaceQu   <- as.numeric(factor(test_data$FireplaceQu  , levels = c("po", "fa","TA","Gd","Ex"), labels = c(1,2,3,4,5)))

#General Encoding

test_data$Foundation   <- as.numeric(factor(test_data$Foundation))
test_data$CentralAir   <- as.numeric(factor(test_data$CentralAir))


# Data Engineering

test_data$Garage <- test_data$GarageCond + test_data$GarageQual
test_data$totalSF<- test_data$TotalBsmtSF+test_data$X1stFlrSF+test_data$X2ndFlrSF
test_data$builtAndRemodled <- test_data$YearBuilt + test_data$YearRemodAdd
test_data$totalBathrooms <- test_data$FullBath*2 + test_data$HalfBath + test_data$BsmtFullBath*2 + test_data$BsmtHalfBath


# Filling nulls after Encoding and Data Engineering

test_data$FireplaceQu[is.na(test_data$FireplaceQu)] <- 0
test_data$ExterQual[is.na(test_data$ExterQual)] <- mean(test_data$ExterQual,na.rm = TRUE)
test_data$HeatingQC[is.na(test_data$HeatingQC)] <- 0
test_data$KitchenQual[is.na(test_data$KitchenQual)] <- 0
test_data$totalSF[is.na(test_data$totalSF)] <- mean(test_data$totalSF,na.rm = TRUE)
test_data$GarageCars[is.na(test_data$GarageCars)] <- 0
test_data$GarageYrBlt[is.na(test_data$GarageYrBlt)] <- mean(test_data$GarageYrBlt,na.rm = TRUE)



# Choosing columns according to the highest correlations with saleprice
test_data <- subset(test_data, select = c('OverallQual', 'builtAndRemodled','totalBathrooms', 'GrLivArea','GarageArea','ExterQual', 'HeatingQC', 
                                          'KitchenQual','TotRmsAbvGrd','Fireplaces','FireplaceQu','totalSF','Foundation','CentralAir','Garage','GarageCars','GarageYrBlt'))
head(test_data)



######### Model Training  ######### 

#random forest 
randomforest = randomForest(formula = SalePrice~.,importance = TRUE, data = data) 
pred_y <- predict(randomforest, newdata = test_data)

SalePrice<-pred_y
submission_df <- data.frame(ID, SalePrice)
write.csv(submission_df, "D:\\NURHAN\\FCIS\\8th Semester\\Distributed Computing\\Project_Data\\FINAL_RandomForest.csv", row.names = FALSE)




############################################################ Model Trials ##################################################################################

###########  1-Linear regression ###########
#regression = lm(SalePrice ~.,  data = data)
#summary(regression)
# prediction
#pred_y <- predict(regression, newdata = test_data, type = "response")


###########  2-SVM ###########
#model = train(SalePrice~., data = data, method = "svmLinear")
# summarising the results
#print(model)
#use model to make predictions on test data
#pred_y = predict(model, test_data)

#model = train(SalePrice~., data = data, method = "radial")
# summarising the results
#print(model)
#use model to make predictions on test data
#pred_y = predict(model, test_data)


########### 3-Tree Model ###########
#tree_model <- rpart(SalePrice~., data = data, method = 'class')
#pred_y <-predict(tree_model, test_data, type = 'class')



############################################################ correlation code sample ############################################################


#dropped_data <- subset(data, select = -c(OverallQual, builtAndRemodled,totalBathrooms, GarageArea, GrLivArea,ExterQual, HeatingQC, 
#                                         KitchenQual,TotRmsAbvGrd,Fireplaces,FireplaceQu,totalSF,Foundation,CentralAir,Garage,GarageCars,GarageYrBlt,GarageCond,GarageQual
#                                         ,TotalBsmtSF,X1stFlrSF,X2ndFlrSF,
#                                         YearBuilt,YearRemodAdd,FullBath,HalfBath,BsmtFullBath,BsmtHalfBath,Id))

#dropped_data <- subset(dropped_data, select = c(SaleCondition,SaleType,YrSold,MoSold,MiscVal,
#                                                MiscFeature,Fence,PavedDrive,GarageFinish,GarageType,
#                                                Functional,KitchenAbvGr,BedroomAbvGr,LowQualFinSF,Electrical
#                                                ,Heating,LotArea,PoolArea,EnclosedPorch,X3SsnPorch,OpenPorchSF
#                                                ,ScreenPorch,WoodDeckSF
#                                                ,SalePrice))

#dropped_data <- mutate_all(dropped_data, as.factor) %>% mutate_all(list(~as.integer(.)))
#dropped_data[is.na(dropped_data)] <- 0


#cor_df <- round(cor(data),2)
#cor_df <- round(cor(dropped_data),2)
#melted_cor <- melt(cor_df)


#heatmap
#ggplot(data = melted_cor, aes(x=Var1, y=Var2, fill=value)) +
#  geom_tile() +
#  geom_text(aes(Var2, Var1, label = value), size = 6) +
#  scale_fill_gradient2(low = "blue", high = "red",
#                       limit = c(-1,1), name="Correlation") +
#  theme(axis.title.x = element_blank(),
#        axis.title.y = element_blank(),
#        panel.background = element_blank())


########################################Visualization rf####################################################


# Get variable importance from the model fit
#ImpData <- as.data.frame(importance(randomforest))
#ImpData$Var.Names <- row.names(ImpData)

#ggplot(ImpData, aes(x=Var.Names, y=`%IncMSE`)) + 
 # geom_segment( aes(x=Var.Names, xend=Var.Names, y=0, yend=`%IncMSE`), color="skyblue") +
  #geom_point(aes(size = IncNodePurity), color="blue", alpha=0.6) +
  #theme_light() +
  #coord_flip() +
  #theme(
   # legend.position="bottom",
    #panel.grid.major.y = element_blank(),
    #panel.border = element_blank(),
    #axis.ticks.y = element_blank()
  #)

