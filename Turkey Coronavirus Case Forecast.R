
## read and part the data
library(dplyr)
data = read.csv("cases_eu.csv")
data_c = data %>%  filter(Type == "Case")
data_d = data %>%  filter(Type == "Death")
data_c$Type = NULL ; data_d$Type = NULL


## Prediction for Coronavirus Cases ##
data_c1 = data_c[1:15,]
data_c2 = data_c[16:30,]


corr = data.frame(cor(data_c1[,unlist(lapply(data_c1, is.numeric))]))
corr

  # stepwise subset selection
start_mod = lm(Turkey~1,data=data_c1)
empty_mod = lm(Turkey~1,data=data_c1)
full_mod = lm(Turkey~.,data=data_c1)
hybridStepwise = step(start_mod,
                      scope=list(upper=full_mod,lower=empty_mod),
                      direction='both')
summary(hybridStepwise)

  # forward selection
start_mod = lm(Turkey~1,data=data_c1)
empty_mod = lm(Turkey~1,data=data_c1)
full_mod = lm(Turkey~.,data=data_c1)
forwardStepwise = step(start_mod,
                       scope=list(upper=full_mod,lower=empty_mod),
                       direction='forward')
summary(forwardStepwise)

    #backward selection
start_mod = lm(Turkey~.,data=data_c1)
empty_mod = lm(Turkey~1,data=data_c1)
full_mod = lm(Turkey~.,data=data_c1)
backwardStepwise = step(start_mod,
                        scope=list(upper=full_mod,lower=empty_mod),
                        direction='backward')
summary(backwardStepwise)

# Linear Regression (backward selection)
modelc_all = lm(Turkey~Europe,
                data=data_c1)
summary(modelc_all)
prediction_all = predict(modelc_all,newdata=data_c2)
prediction_all

prediction_lm_train = predict(modelc_all,newdata=data_c1)
rmse = sqrt(mean((prediction_lm_train-
                    data_c1$Turkey)^2)); rmse

pred_lm = as.data.frame(prediction_all)
names(pred_lm)[names(pred_lm) == "prediction_all"] = "pred_lm"

## Ridge Regression

#regularization
library(caret)
cols_reg = c('Europe',"UK",'Turkey',"France","Italy")
dummies <- dummyVars(Turkey ~ . , data = data_c1[,cols_reg])
train_dummies = predict(dummies, newdata = data_c1[,cols_reg])
pred_dummies = predict(dummies, newdata = data_c2[,cols_reg])

library(glmnet)
x = as.matrix(train_dummies)
y = as.matrix(pred_dummies)
y_train = data_c1$Turkey

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0,
                   family = 'gaussian', lambda = lambdas)

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda


pred_ridge <- predict(ridge_reg, 
                  s = optimal_lambda, newx = y)
pred_ridge

pred_ridge_train <- predict(ridge_reg, 
                      s = optimal_lambda, newx = x)                      
rmse = sqrt(mean((pred_ridge_train-
                    data_c1$Turkey)^2)); rmse


pred_ridge = as.data.frame(pred_ridge)
names(pred_ridge)[names(pred_ridge) == "1"] = "pred_ridge"

## Lasso Regression

lambdas <- 10^seq(2, -3, by = -.1)
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas,
              standardize = TRUE, nfolds = 5)
lambda_best <- lasso_reg$lambda.min 
lambda_best

lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best,
              standardize = TRUE)

pred_lasso <- predict(lasso_model,
                             s = lambda_best, newx = y)
pred_lasso

pred_lasso_train <- predict(lasso_model, 
                            s = optimal_lambda, newx = x)                      
rmse = sqrt(mean((pred_lasso_train-
                    data_c1$Turkey)^2)); rmse


pred_lasso = as.data.frame(pred_lasso)
names(pred_lasso)[names(pred_lasso) == "1"] = "pred_lasso"


### Linear Regression with Dimension Reduction - PCA ###

## read and part the data
library(dplyr)
data = read.csv("cases_eu.csv")
data_c = data %>%  filter(Type == "Case")
data_d = data %>%  filter(Type == "Death")
data_c$Type = NULL ; data_d$Type = NULL

## Prediction for Coronavirus Cases ##
data_c1 = data_c[1:15,]
data_c2 = data_c[16:30,]
data_c2$Turkey = NULL

## Linear Regression with PCA
library(psych);library(prcomp);library(FactoMineR);library(factoextra)
library(corrplot)

trainPredictors = data_c
trainPredictors$Turkey = NULL
KMO(cor(trainPredictors))

pca_facto = PCA(trainPredictors,graph = F)
pca_facto$eig
fviz_eig(pca_facto,ncp=3,addlabels = T)

# moving forward with only one dimension
pca = prcomp(trainPredictors,scale. = T)
trainPredictors = data.frame(pca$x[,1:1])
trainPredictors2 = cbind(trainPredictors[1:15,],data_c1[1:15,]$Turkey)
trainPredictors2 = as.data.frame(trainPredictors2)
# building model
predict_pca_1 = lm(V2~.,trainPredictors2)
summary(predict_pca_1)

# making prediction
trainPredictors3 = trainPredictors[16:30,]
trainPredictors3 = as.data.frame(trainPredictors3)
names(trainPredictors3)[names(trainPredictors3) == "trainPredictors3"] <- "V1"

predict_pca = predict(predict_pca_1,newdata=trainPredictors3)
predict_pca

predict_pca_train = predict(predict_pca_1,newdata=trainPredictors2)
rmse = sqrt(mean((predict_pca_train-
                    trainPredictors2$V2)^2)); rmse

predict_pca = as.data.frame(predict_pca)

### FORECAST VISUALIZATION ###

  # consolidating predictions
dummy_Turkey = data_c1$Turkey
dummy_Turkey = as.data.frame(dummy_Turkey)
names(dummy_Turkey)[names(dummy_Turkey) == "dummy_Turkey"] = "predict_pca"
lm_pca = rbind(dummy_Turkey,predict_pca)
               
dummy_Turkey2 = data_c1$Turkey
dummy_Turkey2 = as.data.frame(dummy_Turkey2)
names(dummy_Turkey2)[names(dummy_Turkey2) == "dummy_Turkey2"] = "pred_lasso"
lm_lasso = rbind(dummy_Turkey2,pred_lasso)

dummy_Turkey3 = data_c1$Turkey
dummy_Turkey3 = as.data.frame(dummy_Turkey3)
names(dummy_Turkey3)[names(dummy_Turkey3) == "dummy_Turkey3"] = "pred_ridge"
lm_ridge = rbind(dummy_Turkey3,pred_ridge)

dummy_Turkey4 = data_c1$Turkey
dummy_Turkey4 = as.data.frame(dummy_Turkey4)
names(dummy_Turkey4)[names(dummy_Turkey4) == "dummy_Turkey4"] = "pred_lm"
lm_backward = rbind(dummy_Turkey4,pred_lm)

days = 1:30
days = as.data.frame(days)

library(tidyverse)
final = cbind(days,lm_pca,lm_lasso,lm_ridge,lm_backward)
final = gather(final,key = "model", value= "value",2:5)

final$col = ifelse(final$days>=15,"2","1")

library(ggplot2)
palette = c("yellow",'#ffa41b','#AEC4EB',
            '#18b0b0','#fe346e',"#1f6650")

ggplot(final,aes(x=days,y=value,color=ifelse(days>=15,model,col)))+
geom_line(size=0.7)+
geom_text(aes(label=ifelse(days>29,as.character(round(value,0)),'')),vjust=0,hjust=0)+
scale_color_manual(values=palette)+
  scale_x_continuous(breaks=seq(0,30,5))+
  scale_y_continuous(breaks=seq(0,60000,10000))+
labs(title = "Turkey Coronavirus Case Prediction",y= "Coronavirus Case", x = "Days")+
labs(colour="Models Used")+
  scale_colour_discrete(labels = c("Actual 15 days", "lasso", "stepwise","ridge","pca"))+
  guides(color = guide_legend(override.aes = list(linetype = 1, size=3)))+
  theme(
  legend.position = "right",
  axis.text = element_text(colour = "white"),
  axis.title.x = element_text(colour = "white", size=rel(1.7)),
  axis.title.y = element_text(colour = "white",size=rel(1.7)),
  panel.background = element_rect(fill="black",colour = "black"),
  panel.grid = element_blank(),
  plot.background = element_rect(fill="black",colour = "black"),
  legend.key = element_rect(fill = "black",colour = "black"),
  legend.background = element_blank(),
  legend.text = element_text(colour="white",size = rel(1)),
  legend.title = element_text(colour="white",size = rel(1)),
  panel.grid.minor = element_line(colour="#202020", size=0.3),
  plot.title = element_text(color="white", size= rel(2),hjust = 0))
                


######## Log Linear Regression ########## (Didn't work well)
data = read.csv("cases_eu.csv")
data
data_c = data %>%  filter(Type == "Case")
data_tr = data_c$Turkey

data_c$Turkey = NULL
data_c$Type = NULL
data_c_log = log(data_c)
data_c_log$Germany = NULL

data_c_log =cbind(data_c_log,data_tr)
names(data_c_log)[names(data_c_log) == "data_tr"] <- "Turkey"


data_c1 = data_c_log[1:14,]
data_c2 = data_c_log[15:30,]
data_c2$Turkey = NULL
data_c1$Germany = NULL

modelc_log = lm(Turkey~.,
                data=data_c1)
summary(modelc_log)
prediction_all = predict(modelc_log,newdata=data_c2)
prediction_all
rmse = sqrt(mean((prediction_all-
                    data_c1$Turkey)^2)); rmse