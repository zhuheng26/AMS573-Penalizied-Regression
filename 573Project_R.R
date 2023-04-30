setwd("~/")
getwd()
newdir <- paste(getwd(),"/Downloads/",sep = "")
setwd(newdir)
getwd()
data.train <- read.csv("train.csv")
data.train <- data.train[,-1]
str(data.train)
data.test <- read.csv("test.csv")
data.test <- data.test[,-1]
str(data.test)

### logistic regression
library(pROC)
glm.logistic <- glm(y ~ ., family = binomial("logit"), data = data.train)
summary(glm.logistic)
coefficients(glm.logistic)

predicted <- predict(glm.logistic, data.test, type="response")
predicted01 <- ifelse(predicted<0.5,0,1)

auc(data.test$y, predicted)
table(data.test$y, predicted01)
auc(data.test$y, predicted01)

### Ridge regression
library(glmnet)
lambda <- 10^seq(10, -2, length=100)
x <- as.matrix(data.train[,1:8])
y <- data.train$y
glm.Ridge <- glmnet(x, y, alpha = 0, lambda = lambda)
plot(glm.Ridge, xvar = "lambda", xlim = c(-5, 5))
legend(x = "topright", legend=colnames(x), col = 1:ncol(x), lty = 1)
title("Ridge Regression",line = 2.5)
abline(h=0,col="red", lty = 2)

set.seed(123)
fit_Ridge = cv.glmnet(x, y, alpha = 0, family = "binomial")
lambda_select = fit_Ridge$lambda.min
lambda_select

# set lambda=lambda_select
glm.Ridge <- glmnet(x, data.train$y, alpha = 0, lambda = 0.0001, thresh = 0.0001)
coef(glm.Ridge)
test <- as.matrix(data.test[,1:8])
predicted <- predict(glm.Ridge, test, type ="response")
predicted01 <- ifelse(predicted<0.5,0,1)

auc(data.test$y, predicted)
table(data.test$y, predicted01)
auc(data.test$y, predicted01)

### Lasso regression
lambda <- 10^seq(10, -2, length=100)
x <- as.matrix(data.train[,1:8])
y <- data.train$y
glm.Lasso <- glmnet(x, data.train$y, alpha = 1, lambda = lambda)
plot(glm.Lasso, xvar = "lambda", xlim = c(-5, 5))
legend(x = "topright", legend=colnames(x), col = 1:ncol(x), lty = 1)
title("Lasso Regression",line = 2.5)
abline(h=0,col="red", lty = 2)

coef(glm.Lasso)

set.seed(123)
fit.Lasso = cv.glmnet(x, y, alpha = 1, family = "binomial")
lambda_select = fit.Lasso$lambda.min
lambda_select

# set lambda=lambda_select
glm.Lasso <- glmnet(x, data.train$y, alpha = 1, lambda = lambda_select)
coef(glm.Lasso)
test <- as.matrix(data.test[,1:8])
predicted <- predict(glm.Lasso, test, type ="response")
predicted01 <- ifelse(predicted<0.5,0,1)

auc(data.test$y, predicted)
table(data.test$y, predicted01)
auc(data.test$y, predicted01)

### Elastic net

# iterate over alpha (0, 0.1, ..., 1), and find the model with minimum cross-validation error as well as the model's corresponding lambda
alpha_set = seq(0,1,0.1)
set.seed(123)
foldid = sample(1:10, size=length(y), replace=TRUE)
mn = Inf
for (alpha in alpha_set){
  fit = cv.glmnet(x, y, foldid=foldid, alpha = alpha)
  curr_res = data.frame(cv_error = fit$cvm, lambda = fit$lambda)
  curr_min_cv_error = curr_res[which(curr_res$cv_error == min(curr_res$cv_error)),]$cv_error
  curr_lambda = curr_res[which(curr_res$cv_error == min(curr_res$cv_error)),]$lambda
  # print(curr_min_cv_error)
  if (curr_min_cv_error < mn){
    res = list(
      min_cv_error = curr_min_cv_error,
      final_lambda = curr_lambda,
      final_alpha = alpha)
  }
}

res

final = glmnet(x, y, foldid=foldid, alpha = res$final_alpha, lambda = res$final_lambda)
coef(final)

test <- as.matrix(data.test[,1:8])
predicted <- predict(final, test, type ="response")
predicted01 <- ifelse(predicted<0.5,0,1)

auc(data.test$y, predicted)
table(data.test$y, predicted01)
auc(data.test$y, predicted01)