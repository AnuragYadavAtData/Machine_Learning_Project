## h1 {

##   text-align: center;

## 

## }


## ----setup, include=FALSE-------------------------------------------------------------------------------
knitr::opts_chunk$set(warning = FALSE,message = FALSE)


## ----load libraries and data, include=FALSE-------------------------------------------------------------
library(caret)
library(VIM)
library(dplyr)
library(kernlab)
library(beepr)
library(caretEnsemble)
library(naniar)
library(psych)
library(corrplot)
library(RANN)

load(file = "allmodelsfinished.RData")
test <- read.csv("test.csv", na.strings="NA", header=TRUE)
train <- read.csv("train.csv", na.strings="NA", header=TRUE)



## ----Structure, echo=FALSE------------------------------------------------------------------------------
str(train)



## ----Change Column Names, include=FALSE-----------------------------------------------------------------
colnames(train)[9] <- "LivingSpace"
colnames(test)[9] <- "LivingSpace"

colnames(train)[10] <- "LotSize"
colnames(test)[10] <- "LotSize"

colnames(train)[11] <- "Pool"
colnames(test)[11] <- "Pool"




## ----format data ,echo=FALSE----------------------------------------------------------------------------
t <- train[-1]

t$sent <- as.factor(t$sent)
t$reno <- as.factor(t$reno)
t$year <- factor(t$year,order = T, levels = c('1','2','3','4','5','6','7','8','9','10','11','12'))
t$built <- factor(t$built,order = T,levels = c('1','2','3','4','5','6','7','8','9','10'))

levels(t$sent) <- c("No_Sat","Sat")
levels(t$reno) <- c("No_Ren","Ren")



## ----Data Split, include=FALSE--------------------------------------------------------------------------
set.seed(42)
rows <- sample(nrow(t))
t <- t[rows,]

split <- round(nrow(t) * 0.8)

train_t <- t[1:split,]
test_t <- t[(split+1):nrow(t),]


## ----Confirm Split, echo=TRUE---------------------------------------------------------------------------
nrow(train_t)/nrow(test_t)


## ----echo=FALSE, cache=TRUE-----------------------------------------------------------------------------
gg_miss_var(t, facet = sent)


## ----echo=FALSE, cache=TRUE-----------------------------------------------------------------------------
vis_miss(train_t)



## ----imputation, include=FALSE--------------------------------------------------------------------------
train_t %>% aggr(combined = T, numbers = T)



vars_by_NAs <- train_t %>% is.na() %>% colSums() %>% sort(decreasing = F) %>% names()
set.seed(42)
train_imp <- train_t %>% select(vars_by_NAs) %>% kNN(k= 5)


test_t %>% aggr(combined = T, numbers = T)



vars_by_NAs <- test_t %>% is.na() %>% colSums() %>% sort(decreasing = F) %>% names()
set.seed(42)
test_imp <- test_t %>% select(vars_by_NAs) %>% kNN(k= 5)

train_imp <- train_imp %>% select(1:14)
test_imp <- test_imp %>% select(1:14)



## ----summary, echo=FALSE,cache=TRUE---------------------------------------------------------------------
#
describe(train_imp)



## ----echo=FALSE-----------------------------------------------------------------------------------------
dat_s <- subset(train_imp, select = c("lon", "LotSize"))
trans <- preProcess(dat_s, method = c("BoxCox"))
transformed <- predict(trans, dat_s)

par(mfrow = c(1, 2), oma = c(2, 2, 2, 2))
hist(train_t$LotSize, main = "Before Transformation", 
    xlab = "LotSize")
hist(transformed$LotSize, main = "After Transformation", 
    xlab = "LotSize")

train_imp$lon <- transformed$lon
train_imp$LotSize <- transformed$LotSize


## ----correlation, echo=FALSE, cache=TRUE----------------------------------------------------------------
nums <- unlist(lapply(train_imp, is.numeric), use.names = FALSE)  
nums_t <- train_imp[ , nums]
corrplot(cor(nums_t))



## ----scale----------------------------------------------------------------------------------------------

rg_train <- train_imp[-2]
rg_test <- test_imp[-2]

cm_train <- train_imp[-1]
cm_test <- test_imp[-1]

preProcValues_r <- preProcess(rg_train, method = c("center", "scale"))
trainTransformed_r <- predict(preProcValues_r, rg_train)
testTransformed_r <- predict(preProcValues_r, rg_test)

preProcValues <- preProcess(cm_train, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, cm_train)
testTransformed <- predict(preProcValues, cm_test)



## ----class imbalance, echo=FALSE------------------------------------------------------------------------
summary(train_imp$sent)



## ----traincontrolClassification, include=FALSE----------------------------------------------------------
ctrl_none <- trainControl(method="cv",number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
ctrl_up <- trainControl(method="cv",number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,sampling = "up")




## ----include=FALSE--------------------------------------------------------------------------------------
set.seed(42)
knnfit_none <- train(sent ~ . ,trainTransformed, method = "knn",trControl = ctrl_none)



## ----include=FALSE--------------------------------------------------------------------------------------
plot(knnfit_none)
varImp(knnfit_none)


## ----echo=FALSE-----------------------------------------------------------------------------------------
knnPredict <- predict(knnfit_none,newdata = testTransformed)
confusionMatrix(knnPredict, testTransformed$sent, )


## ----include=FALSE--------------------------------------------------------------------------------------
# set.seed(42)
# glmnetfit <- train(sent ~ . ,trainTransformed, method = "glmnet",trControl = ctrl_none, tuneLength = 10)
# beep()



## ----include=FALSE--------------------------------------------------------------------------------------
plot(glmnetfit)
varImp(glmnetfit)



## ----echo=FALSE-----------------------------------------------------------------------------------------
Predict <- predict(glmnetfit,newdata = testTransformed)
confusionMatrix(Predict, testTransformed$sent, )


## ----cache=FALSE, include=FALSE-------------------------------------------------------------------------
set.seed(42)
svm <- train(sent ~ . ,trainTransformed, method = "svmLinear",trControl = ctrl_none)
beep()




## ----include=FALSE--------------------------------------------------------------------------------------
varImp(svm)


## ----echo=FALSE-----------------------------------------------------------------------------------------
Predict <- predict(svm,newdata = testTransformed)
confusionMatrix(Predict, testTransformed$sent, )


## ----include=FALSE--------------------------------------------------------------------------------------
set.seed(42)
decisiontTree <- train(sent ~ . ,trainTransformed, method = "rpart",trControl = ctrl_up,metric = "Sens")
beep()




## ----include=FALSE--------------------------------------------------------------------------------------
plot(decisiontTree)
varImp(decisiontTree)


## -------------------------------------------------------------------------------------------------------
Predict <- predict(decisiontTree,newdata = testTransformed)
confusionMatrix(Predict, testTransformed$sent, )


## ----include=FALSE--------------------------------------------------------------------------------------
set.seed(42)
Rf_hyp_up <- train(sent ~ . ,trainTransformed, method = "ranger",trControl = ctrl_up,tuneLength = 10)
beep()



## ----include=FALSE--------------------------------------------------------------------------------------
plot(Rf_hyp_up)


## ----echo=FALSE-----------------------------------------------------------------------------------------
Predict <- predict(Rf_hyp_up,newdata = testTransformed)
confusionMatrix(Predict, testTransformed$sent, )


## ----echo=TRUE------------------------------------------------------------------------------------------
preProcValues_lda <- preProcess(trainTransformed, method = "pca", pcaComp = 2)
trainTransformed_lda <- predict(preProcValues_lda, trainTransformed)
trainTransformed_lda <- trainTransformed_lda[-2]
trainTransformed_lda <-trainTransformed_lda[-2]
trainTransformed_lda <-trainTransformed_lda[-2]

preProcValues_lda_2 <- preProcess(testTransformed, method = "pca", pcaComp = 2)
trainTransformed_lda_2 <- predict(preProcValues_lda_2, testTransformed)
trainTransformed_lda_2 <- trainTransformed_lda_2[-2]
trainTransformed_lda_2 <-trainTransformed_lda_2[-2]
trainTransformed_lda_2 <-trainTransformed_lda_2[-2]



## ----echo=FALSE-----------------------------------------------------------------------------------------

set.seed(42)
lda.fit = train(sent ~ ., data=trainTransformed_lda, method="lda",
                trControl = ctrl_none)

predict.test <- predict(lda.fit,trainTransformed_lda_2)

confusionMatrix(predict.test,testTransformed$sent)

library(MASS)
linear <- lda(sent~., trainTransformed_lda)
p <- predict(linear, trainTransformed_lda_2)

p <- as.data.frame(p)
p <- cbind(p,testTransformed$sent)

p %>%
  filter(testTransformed$sent %in% c("No_Sat","Sat")) %>%
  ggplot( aes(x=LD1, color=testTransformed$sent, fill=testTransformed$sent)) +
    geom_density(alpha=0.6)




## -------------------------------------------------------------------------------------------------------
model_list <- list(
  knn = knnfit_none,
  glmnet = glmnetfit,
  svmLinear = svm,
  rpart = decisiontTree,
  ranger = Rf_hyp_up,
  lda = lda.fit
)
resamps <- resamples(model_list)

bwplot(resamps, metric = "ROC")

densityplot(resamps,metric = "Spec",auto.key=TRUE)


## ----include=FALSE--------------------------------------------------------------------------------------
ctrl <- trainControl(method="cv",number = 5)


## ----include=FALSE--------------------------------------------------------------------------------------
set.seed(42)
LR_step <- train(price ~ . ,trainTransformed_r, method = "glmStepAIC",direction ="backward",trControl = ctrl)
beep()



## ----eval=FALSE, include=FALSE--------------------------------------------------------------------------
## plot(LR_step$finalModel)
## varImp(LR_step)


## ----include=FALSE--------------------------------------------------------------------------------------
knnPredict <- predict(LR_step,newdata = testTransformed_r)
error <- knnPredict - testTransformed_r[["price"]]
MSE <- (mean(error^2))
MSE


## ----echo=FALSE-----------------------------------------------------------------------------------------
MSE # MSE


## ----include=FALSE--------------------------------------------------------------------------------------

lambda_grid <- seq(0, 3, 0.1)
alpha_grid <- seq(0, 1, 0.1)
srchGrid <- expand.grid(.alpha = alpha_grid, .lambda = lambda_grid)
set.seed(42)
RR <- train(price ~ . ,trainTransformed_r, method = "glmnet",tuneGrid = srchGrid,trControl = ctrl)
beep()




## ----include=FALSE--------------------------------------------------------------------------------------
knnPredict <- predict(RR,newdata = testTransformed_r)
error <- knnPredict - testTransformed_r[["price"]]
MSE <- (mean(error^2))
MSE


## ----echo=FALSE-----------------------------------------------------------------------------------------
MSE # MSE


## ----cache=FALSE, include=FALSE-------------------------------------------------------------------------
set.seed(42)
rf_r <- train(price ~ . ,trainTransformed_r, method = "ranger",trControl = ctrl)
beep()



## ----echo=FALSE-----------------------------------------------------------------------------------------
Predict <- predict(rf_r,newdata = testTransformed_r)
error <- Predict - testTransformed_r[["price"]]
MSE <- (mean(error^2))
MSE



## ----cache=FALSE, include=FALSE-------------------------------------------------------------------------
set.seed(42)
xgb <- train(price ~ . ,trainTransformed_r, method = "xgbTree",trControl = ctrl)
beep()




## ----include=FALSE--------------------------------------------------------------------------------------
varImp(xgb)


## ----echo=FALSE-----------------------------------------------------------------------------------------
knnPredict <- predict(xgb,newdata = testTransformed_r)
error <- knnPredict - testTransformed_r[["price"]]
MSE <- (mean(error^2))
MSE


## ----echo=FALSE, cache=FALSE----------------------------------------------------------------------------
# full model use this
ndataFull <- rbind(trainTransformed_r,testTransformed_r)
set.seed(42)
brnn <- train(price ~ . ,ndataFull, method = "brnn",trControl = ctrl )
beep()

knnPredict <- predict(brnn,newdata = testTransformed_r)
error <- knnPredict - testTransformed_r[["price"]]
MSE <- (mean(error^2))
MSE
brnn_1 <- brnn


## ----echo=FALSE-----------------------------------------------------------------------------------------

model_list <- list(
  glmStepAIC = LR_step,
  glmnet = RR,
  ranger = rf_r,
  xgbTree = xgb,
  brnn = brnn_1
)
resamps <- resamples(model_list)

dotplot(resamps,metric = "RMSE")
dotplot(resamples(model_list))

