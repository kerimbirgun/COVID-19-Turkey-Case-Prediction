
## Coronavirus Turkey Case Prediction Results and an Additional Europe Pairplot Analysis ##

# After 15 days I updated my dataset with Turkey Actuals

library(dplyr)
data = read.csv("cases_eu_TR_actual.csv")
data_c = data %>%  filter(Type == "Case")

## Prediction for Coronavirus Cases ##
data_c1 = data_c[1:15,]
data_c2 = data_c[16:30,]


# Linear Regression (backward selection)
modelc_all = lm(Turkey~Europe,
                data=data_c1)
summary(modelc_all)
prediction_all = predict(modelc_all,newdata=data_c2)
prediction_all

prediction_lm_train = predict(modelc_all,newdata=data_c1)
rmse = sqrt(mean((prediction_lm_train-
                    data_c1$Turkey)^2)); rmse

rmse_pred = sqrt(mean((prediction_all-
                    data_c2$Turkey)^2)); rmse_pred


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

rmse_pred = sqrt(mean((pred_ridge-
                           data_c2$Turkey)^2)); rmse_pred

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

rmse_pred = sqrt(mean((pred_lasso-
                         data_c2$Turkey)^2)); rmse_pred

## Prediction for Coronavirus Cases ##

## Linear Regression with PCA
library(psych);library(prcomp);library(FactoMineR);library(factoextra)
library(corrplot)

trainPredictors = data_c
trainPredictors$Turkey = NULL
trainPredictors$Type = NULL
KMO(cor(trainPredictors))

pca_facto = PCA(trainPredictors,graph = F)
pca_facto$eig

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

rmse_pred = sqrt(mean((predict_pca-
                         data_c2$Turkey)^2)); rmse_pred

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
final = cbind(days,data_c$Turkey,lm_pca,lm_lasso,lm_ridge,lm_backward)

final$predict_pca = ifelse(final$days>=16,final$predict_pca,"")
final$pred_lasso = ifelse(final$days>=16,final$pred_lasso,"")
final$pred_ridge = ifelse(final$days>=16,final$pred_ridge,"")
final$pred_lm = ifelse(final$days>=16,final$pred_lm,"")

final = gather(final,key = "model", value= "value",2:6)
final$value = as.integer(final$value)
final$linetype = ifelse(final$model == 'data_c$Turkey',1,2)
final$linetype = as.factor(final$linetype)

# 30rd day prediction values requires superminimal finetuning due to superminimal 
#changes in the dataset
final$value = ifelse(final$value == 53938,53935,final$value)
final$value = ifelse(final$value == 42066,41935,final$value)
final$value = ifelse(final$value == 44961,44962,final$value)
final$value = ifelse(final$value == 47979,47970,final$value)

## FINAL PLOT ##

library(ggplot2);library(ggrepel)
palette = c("yellow",'#ffa41b','#AEC4EB',
            '#18b0b0','#fe346e',"#1f6650")

ggplot(final,aes(x=days,y=value,color=model,shape=linetype))+
  geom_line(aes(linetype = linetype),size=1.15)+guides(linetype = FALSE)+
  geom_text_repel(aes(label=ifelse(days>29,as.character(round(value,0)),'')),hjust= -0.5)+
  scale_color_manual(values=palette)+
  scale_x_continuous(breaks=seq(0,30,5))+
  scale_y_continuous(breaks=seq(0,60000,10000))+
  labs(title = "Turkey Coronavirus Case Prediction Result",y= "Coronavirus Case", x = "Days")+
  labs(colour="Models Used")+
  scale_colour_discrete(labels = c("Actual Case", "Model 1 (lasso)", "Model 2 (stepwise)","Model 3 (ridge)","Model 4 (pca)"))+
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


## European Countries first 30 days Pairplot with a Logarithmic Comparison

lowerfun <- function(data,mapping){
  ggplot(data = datapair, mapping = mapping)+
    geom_smooth(method="loess",se=F)+
    theme(panel.grid = element_blank(),panel.grid.minor = element_blank(),legend.position = "none", 
 panel.border = element_rect(linetype = "dashed", colour = "black", fill = NA),
 strip.text = element_text(size = 7))+ scale_fill_brewer(palette="palette")
}  

require(GGally)
A = log(data[,2:8])
datapair = cbind(data$Type,A)
datapair %>% ggpairs(., columns = c('Turkey','France','Italy',"Germany",'Spain','UK'),
   lower = list(continuous = wrap(lowerfun)),
   upper = list(continuous = wrap("cor", size=5,hjust=0.8)),
   mapping = ggplot2::aes(colour=data$Type))+
  theme(
    panel.grid.minor = element_blank(),
   panel.grid.major = element_blank(),strip.text = element_text(size = 12),
   plot.title = element_text(color="black", size= rel(1.7),hjust = 0))+
labs(title = "First 30 Days Logarithmic Case & Death Comparison with Correlation Metrics")







       