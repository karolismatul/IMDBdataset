######################################
#                                    #
# IMDB Classification and Regression #
#                                    #
######################################
##           ### MAY 2017 ###       ##
######################################

## Loading Libraries ##
library("rvest")
library("purrr")
library(readr)
library(stringr)
library(stringi)
library(tidyr)
library(httr)
library(aod)
library(ggplot2)
library(ggthemes)
library(XML)
library(glmnet)
library(dplyr)
library(glmnet)
library(leaps)
library(rio)
library(Amelia)
require(parallel)
require(data.table)
require(plyr)
library(DAAG)
library(caret)
library(plotly)
library(MASS)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(rpart)
library(randomForest)
library(hydroGOF)
library(e1071)

### Setting seed for reproductability
set.seed(1337)   

### Forcing R to not display scientific notation####
options("scipen"=10)

## Importing database

moviedatabase <- read.csv('https://raw.githubusercontent.com/karolismatul/IMDBdataset/master/moviedataadjusted15.csv',header=T,stringsAsFactors = F)

## Checking for missing values

missmap(moviedatabase, main = "Missing Value Overview")

## Changing several values to factors ##

moviedatabase$winner <- as.factor(moviedatabase$winner)
moviedatabase$star_both <- as.factor(moviedatabase$star_both)
moviedatabase$star_actor <- as.factor(moviedatabase$star_actor)
moviedatabase$star_director <- as.factor(moviedatabase$star_director)

## Check if it worked ##
summary(moviedatabase)

####Splitting Data into training and test sets######
smp_size <- floor(0.75 * nrow(moviedatabase))
train_ind <- sample(seq_len(nrow(moviedatabase)), size = smp_size)

train <- moviedatabase[train_ind, ]
test <- moviedatabase[-train_ind, ]

####------------PREDICTION--------------------------####

### 1. Simple Linear Regression for Benchmark ####

##Linear Regression (With one predictor variable)##
##We will use this as a benchmark for how well our predictions do##
basefit <- lm(gross ~ budget, data=train)
summary(basefit) # show results MRS of 0.4375 with adjusted dataset 0.6309

linearpredtest <- predict(basefit, test)
rmse(test$gross, linearpredtest) #RMSE OF 46746634 with adjusted dataset 43934631

linearpredtrain <- predict(basefit, train)

rmse(train$gross, linearpredtrain) #RMSE OF 53224047 with adjusted dataset 41316053

### GRAPHING OUTLIERS IN THE DATA ####
par(mfrow=c(1, 2))  # divide graph area in 2 columns
boxplot(train$budget, main="Budget", sub=paste("Outlier rows: ", boxplot.stats(train$budget)$out))  # box plot for 'speed'
boxplot(train$gross, main="Gross", sub=paste("Outlier rows: ", boxplot.stats(train$gross)$out))  # box plot for 'distance'

cor(train$budget, train$gross)  # calculate correlation between budget and gross 0.6614478

AIC(basefit)  # AIC => 90334.99  // 63514.96
BIC(basefit)  # BIC => 90352.29  // 63531.22

## Checking the RMSE

### 2. Multiple Linear Regression (With many predictor variables)
multifit <- lm(gross ~ star_director + title_year + duration + star_both + star_actor + budget + actor_1_facebook_likes + actor_2_facebook_likes + actor_3_facebook_likes + cast_total_facebook_likes + director_facebook_likes + facenumber_in_poster + content_rating, data=train)
summary(multifit) # show results MRS of 0.5023   /// 0.6593

## Checking the RMSE
multipredtest <- predict(multifit, test)
rmse(test$gross, multipredtest) #RMSE of 54559168 /// 42899117

multipredtrain <- predict(multifit, train)
rmse(train$gross, multipredtrain) #RMSE of 48821310 /// 40261147

AIC(multifit)  # AIC => 90230.26 //// 63419.35
BIC(multifit)  # BIC => 90368.63 //// 63538.62

### 3. Making a Classification Tree ###

fit2 <- rpart(winner ~ star_director + title_year + duration + star_both + star_actor + budget + director_facebook_likes + cast_total_facebook_likes + facenumber_in_poster + content_rating, data=train, method="class")
##viewresults
printcp(fit2)
##view tree
fancyRpartPlot(fit2)
##check accuracy
t_pred = predict(fit2,test,type="class")
confMat <- table(test$winner,t_pred)
accuracyclass <- sum(diag(confMat))/sum(confMat)
print(accuracyclass) ## Accuracy of 0.6362484% //// 0.7007168%

### 4. Making a RandomForrest          ####

fit3 <- randomForest(as.factor(winner) ~ star_director + title_year + duration + star_both + star_actor + budget + cast_total_facebook_likes + director_facebook_likes + facenumber_in_poster,
                     data=train, 
                     importance=TRUE, 
                     ntree=1000,
                     do.trace=T)
varImpPlot(fit3)
fit3$confusion[, 'class.error']

predict(fit3, data=test)-> prediction2

postResample(prediction2, test$winner)  ## 0.5386565% Accuracy // Adjusted= 0.6308244%

### --------------- Visualizing the Predictions ---------- ###

# Setting better View --- Four graphs per page layout --- ##
layout(matrix(c(1,2,3,4),2,2))
## 1. Diagnostic Analysis Plots ###
# diagnostic plots for basefit 
plot(basefit)

# diagnostic plots multifit
plot(multifit)


### 2. Graphing the Differences between them ###

plot(predict(basefit),train$gross, main="Linear Regression",
     xlab="Predicted Gross",ylab="Actual Gross")
abline(a=0,b=1, col='red')

plot(predict(multifit),train$gross, main="Multi-Linear Regression",
     xlab="Predicted Gross",ylab="Actual Gross")
abline(a=0,b=1, col='red')







####---PLOTTING RESIDUALS ON LINEAR REGRESSION BASE MODEL-----####
d <- train
fit <- lm(gross ~ budget, data=d)

d$predicted <- predict(fit)   # Save the predicted values
d$residuals <- residuals(fit) # Save the residual values

ggplot(d, aes(x = budget, y = gross)) +
  ggtitle("Residuals of Simple Linear Regression") +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +
  geom_segment(aes(xend = budget, yend = predicted), alpha = .2) +
  geom_point(aes(color = residuals)) +  # Color mapped here
  scale_color_gradient2(low = "blue", mid = "white", high = "red") +  # Colors to use here
  guides(color = FALSE) +
  ##coord_cartesian(xlim=c(0,300000000), ylim=c(0,300000000)) +
  geom_point(aes(y = predicted), shape = 1) +
  theme_bw()


###############5. GRAPHING RESULTS ##########


### IMDB SCORE VS ROI####

ggplot(moviedatabase, aes(x = gross, y = roi_ratio)) + 
  geom_point() +
  theme_economist() +
  stat_smooth(method = "lm", col = "red")

#### GROSS VS BUDGET ####
ggplot(moviedatabase, aes(x = budget, y = gross)) + 
  geom_point() + 
  ggtitle("Movie Budget over Gross") +
  geom_smooth() + 
  theme_economist() + 
  labs( x = "Budget",
        y = "Gross")

#### GROSS VS ROI RATIO ####
ggplot(moviedatabase, aes(x = gross, y = roi_ratio)) + 
  geom_point() + 
  geom_smooth() + 
  theme_economist() + 
  labs( x = "Gross",
        y = "Roi Ratio")

### GROSS VS TITLE YEAR ####

ggplot(moviedatabase, aes(x = title_year, y = roi_ratio)) + 
  geom_point() +
  ggtitle("ROI over the Years") +
  geom_smooth() + 
  theme_economist() + 
  labs( x = "Title Year",
        y = "ROI")

#### BUDGET VS TITLE YEAR ###

ggplot(moviedatabase, aes(x = title_year, y = budget)) + 
  geom_point() + 
  geom_smooth() + 
  theme_economist() + 
  labs( x = "Title Year",
        y = "Budget")

#### GROSS VS IMDB SCORE ####
ggplot(moviedatabase, aes(x = gross, y = imdb_score)) + 
  geom_point() + 
  geom_smooth() + 
  theme_economist() + 
  labs( x = "Gross ($)",
        y = "IMDB Score")

#### BUDGET VS TITLE YEAR ###

ggplot(moviedatabase, aes(x = title_year, y = imdb_score)) + 
  geom_point() + 
  geom_smooth() + 
  theme_economist() + 
  labs( x = "Title Year",
        y = "IMDB Score")