## AMI22T Home Exercise 2, Problem 1
## Patient with Parkinson's Disease or Healthy?
## Saumya Gupta, DS


# set path of all-data export import directory
setwd('C:/Users/gupta/OneDrive/Documents/MS-DS/AMI22T/HomeExercises/HomeExercise2/')


# load required packages
library(prettyR)
library(data.table)
library(dplyr)
library(tree)
library(MLmetrics)
library(glmnet)
library(e1071)
library(randomForest)
library(gbm)
library(ggplot2)


# Utility Function ----
mode_aggregate <- function(class, id) {
  return (aggregate(list(class = class),
                    by = list(id = id),
                    FUN = Mode)$class %>%
            as.factor())
}


# Data Read ----
# load data and store in data table
pd.speech.features <-
  fread("pd_speech_features.csv", skip = 1, header = T)

pd.speech.features <- data.frame(pd.speech.features)


# set seed
set.seed(9899)


# check for columns with missing data
names(which(colSums(is.na(pd.speech.features)) > 0))


# check for any non-numeric variables
sapply(pd.speech.features[c(names(pd.speech.features)[sapply(pd.speech.features,
                                                             class) != 'numeric'])],
       class)


# check gender distribution
ggplot(pd.speech.features,
       aes(x = as.factor(class),
           fill = as.factor(gender))) +
  geom_bar(position = "dodge")


# Normalization ----
# exclude id, gender and class variables from scaling process
# exclude app_entropy_shannon_10_coef (integer64) and scale it separately
# (otherwise transforms latter to NA!)
pd.speech.features[,-c(1, 2, 201, 755)] <-
  scale(pd.speech.features[,-c(1, 2, 201, 755)])

pd.speech.features$app_entropy_shannon_10_coef <-
  scale(pd.speech.features$app_entropy_shannon_10_coef)


# check for presence of highly correlated variables
correlations <- cor(as.matrix(pd.speech.features))

correlations <- as.data.frame(correlations)

correlations[correlations < 0.8 | correlations == 1] <- ""

correlations <-
  correlations[!sapply(correlations, function(x)
    all(is.na(x) | x == ""))]


# Train-Test Split ----
# 80% of humans go for training
train <- as.vector(sample(0:251, floor(252 * 0.8)))

features.train <-
  pd.speech.features[pd.speech.features$id %in% train, ]

features.test <-
  pd.speech.features[!(pd.speech.features$id %in% train), ]


# Task 1 ----
## Modelling ----
data.train <- features.train %>%
  select(-c("id"))

data.train$class <- as.factor(data.train$class)

true.class <- as.vector(features.test$class)

# get mode aggregates for majority voting
true.class <- mode_aggregate(true.class, features.test$id)

### Decision Tree ----
# fit
tree.features <- tree(class ~ ., data.train)

summary(tree.features)

plot(tree.features)
text(tree.features, pretty = 0)

# predict for test
tree.pred <- predict(tree.features, features.test, type = "class")

# get mode aggregates
tree.pred <- mode_aggregate(tree.pred, features.test$id)

# evaluate prediction
prop.table(table(tree.pred, true.class), margin = 2)

acc.dt <-
  (table(tree.pred,
         true.class)[1] + table(tree.pred,
                                true.class)[4]) / (nrow(features.test) /
                                                     3)

f1score.dt <- F1_Score(true.class, tree.pred)

# perform pruning with cross-validation and let it make feature selection
cv.features <- cv.tree(tree.features)

# find best number of terminal nodes
plot(cv.features$size, cv.features$dev, type = "b")

# fit with best number of terminal nodes
prune.features <- prune.tree(tree.features, best = 4)

summary(prune.features)

plot(prune.features)
text(prune.features, pretty = 0)

# predict for test
prune.tree.pred <-
  predict(prune.features, features.test, type = "class")

# get mode aggregates
prune.tree.pred <- mode_aggregate(prune.tree.pred, features.test$id)

# evaluate prediction
prop.table(table(prune.tree.pred, true.class), margin = 2)

acc.dt.cv <-
  (table(prune.tree.pred,
         true.class)[1] + table(prune.tree.pred,
                                true.class)[4]) / (nrow(features.test) /
                                                     3)

f1score.dt.cv <- F1_Score(true.class, prune.tree.pred)

### Support-Vector Machine ----
# perform feature selection using lasso regression
glm.fit <-
  cv.glmnet(
    x = model.matrix(class ~ ., data = data.train)[,-1],
    y = as.numeric(data.train$class),
    type.measure = 'deviance',
    nfolds = 10,
    alpha = 0.5
  )

# get all coefficients
coefficients <- coef(glm.fit, s = 'lambda.1se', exact = TRUE)

# filter out coefficients equal to 0
important.variables.lasso <-
  row.names(coefficients)[which(coefficients != 0)]

# get significant variables
important.variables.lasso <-
  important.variables.lasso[!(important.variables.lasso %in%
                                '(Intercept)')]

# linear kernel
# tune cost hyperparameter
tune.out <- tune(
  svm,
  class ~ .,
  data = data.train %>% select(c(
    all_of(important.variables.lasso), "class"
  )),
  kernel = "linear",
  ranges = list(cost = c(0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100))
)

summary(tune.out)

# use linear kernel with optimal cost
svm.l.features <- svm(
  class ~ .,
  data = data.train %>% select(c(
    all_of(important.variables.lasso), "class"
  )),
  kernel = "linear",
  cost = 0.1,
  scale = FALSE
)

summary(svm.l.features)

# predict for test
svm.l.pred <- predict(svm.l.features, features.test)

# get mode aggregates
svm.l.pred <- mode_aggregate(svm.l.pred, features.test$id)

# evaluate prediction
prop.table(table(predict = svm.l.pred, truth = true.class), margin = 2)

acc.svm.l <-
  (table(svm.l.pred,
         true.class)[1] + table(svm.l.pred,
                                true.class)[4]) / (nrow(features.test) /
                                                     3)

f1score.svm.l <- F1_Score(true.class, svm.l.pred)

# radial kernel
# tune cost and gamma hyperparameter
tune.out <- tune(
  svm,
  class ~ .,
  data = data.train %>% select(c(
    all_of(important.variables.lasso), "class"
  )),
  kernel = "radial",
  ranges = list(
    cost = c(1, 5, 10, 70, 80, 90, 100),
    gamma = c(0.01, 0.1)
  )
)

summary(tune.out)

# use radial kernel with optimal cost and gamma
svm.r.features <- svm(
  class ~ .,
  data = data.train %>% select(c(
    all_of(important.variables.lasso), "class"
  )),
  kernel = "radial",
  cost = 5,
  gamma = 0.01,
  scale = FALSE
)

summary(svm.r.features)

# predict for test
svm.r.pred <- predict(svm.r.features, features.test)

# get mode aggregates
svm.r.pred <- mode_aggregate(svm.r.pred, features.test$id)

# evaluate prediction
prop.table(table(predict = svm.r.pred, truth = true.class), margin = 2)

acc.svm.r <-
  (table(svm.r.pred,
         true.class)[1] + table(svm.r.pred,
                                true.class)[4]) / (nrow(features.test) /
                                                     3)

f1score.svm.r <- F1_Score(true.class, svm.r.pred)

### Bagging ----
# perform feature selection by finding variables with high mean drop in accuracy
rf.features.selection <-
  randomForest(class ~ ., data = data.train, importance = TRUE)

varImpPlot(rf.features.selection)

# get RF filter variables
important.variables.rf <-
  data.frame(importance(rf.features.selection)) %>%
  slice_max(order_by = MeanDecreaseAccuracy, n = 25) %>%
  rownames()

# find optimal ntree value using OOB error estimates
plot(rf.features.selection)

# fit with optimal ntree value and selected variables
bag.features <- randomForest(
  class ~ .,
  data = data.train %>% select(c(all_of(
    important.variables.rf
  ), "class")),
  mtry = 25,
  ntree = 50
)

# predict for test
bag.pred <- predict(bag.features, newdata = features.test)

# get mode aggregates
bag.pred <- mode_aggregate(bag.pred, features.test$id)

# evaluate prediction
prop.table(table(bag.pred, true.class), margin = 2)

acc.bag <-
  (table(bag.pred, true.class)[1] + table(bag.pred,
                                          true.class)[4]) / (nrow(features.test) /
                                                               3)

f1score.bag <- F1_Score(true.class, bag.pred)

### Random Forest ----
# find optimal mtry value using 500 trees for tuning and previously selected features
mtry <- tuneRF(
  data.train %>% select(c(all_of(
    important.variables.rf
  ))),
  data.train$class,
  ntreeTry = 500,
  stepFactor = 1.5,
  improve = 0.01,
  trace = TRUE,
  plot = TRUE
)

print(mtry)

# fit with optimal mtry value and previously selected features
# use optimal ntree value found during bagging
rf.features <- randomForest(
  class ~ .,
  data = data.train %>% select(c(all_of(
    important.variables.rf
  ), "class")),
  mtry = 7,
  ntree = 50
)

# predict for test
rf.pred <- predict(rf.features, newdata = features.test)

# get mode aggregates
rf.pred <- mode_aggregate(rf.pred, features.test$id)

# evaluate prediction
prop.table(table(rf.pred, true.class), margin = 2)

acc.rf <-
  (table(rf.pred, true.class)[1] + table(rf.pred,
                                         true.class)[4]) / (nrow(features.test) /
                                                              3)

f1score.rf <- F1_Score(true.class, rf.pred)

### Boosting ----
# perform feature selection by finding variables with high relative influence
# boosting using 1000 trees, 0.01 shrinkage and default depth
boost.features.selection <- gbm(
  as.integer(class) - 1 ~ .,
  data = data.train,
  n.trees = 1000,
  shrinkage = 0.01,
  distribution = "bernoulli"
)

# get Boosting filter variables
important.variables.boost <-
  summary(boost.features.selection) %>%
  slice_max(order_by = rel.inf, n = 25) %>%
  rownames()

# use the selected variables
boost.features <- gbm(
  as.integer(class) - 1 ~ .,
  data = data.train %>% select(c(
    all_of(important.variables.boost), "class"
  )),
  n.trees = 1000,
  shrinkage = 0.01,
  distribution = "bernoulli"
)

# predict for test
boost.pred <- predict(boost.features,
                      newdata = features.test,
                      n.trees = 1000,
                      type = "response")

boost.binary.pred <- as.factor(ifelse(boost.pred > 0.5, 1, 0))

# get mode aggregates
boost.binary.pred <-
  mode_aggregate(boost.binary.pred, features.test$id)

# evaluate prediction
prop.table(table(boost.binary.pred, true.class), margin = 1)

acc.boost <-
  (table(boost.binary.pred,
         true.class)[1] + table(boost.binary.pred,
                                true.class)[4]) / (nrow(features.test) /
                                                     3)

f1score.boost <- F1_Score(true.class, boost.binary.pred)

## Comparison Results ----
# create a data frame for comparison
accuracies <-
  data.frame(
    Model = c(
      "Decision Tree",
      "Decision Tree (Pruned)",
      "Support-Vector Machine (Linear)",
      "Support-Vector Machine (Radial)",
      "Bagging",
      "Random Forest",
      "Boosting"
    ),
    Accuracy = c(
      acc.dt,
      acc.dt.cv,
      acc.svm.l,
      acc.svm.r,
      acc.bag,
      acc.rf,
      acc.boost
    ),
    F1Score = c(
      f1score.dt,
      f1score.dt.cv,
      f1score.svm.l,
      f1score.svm.r,
      f1score.bag,
      f1score.rf,
      f1score.boost
    )
  )

accuracies <-
  accuracies %>% mutate_if(is.numeric, round, digits = 2)

# plot all model results
ggplot(accuracies, aes(Model, Accuracy, fill = Model)) +
  geom_bar(stat = "identity")

ggplot(accuracies, aes(Model, F1Score, fill = Model)) +
  geom_bar(stat = "identity")


# Task 2 ----
## Dimensionality Reduction (Using PCA) ----
# find principal components for train
principal.components <-
  prcomp(features.train %>% select(-c("id", "class")))

# calculate variances using standard deviation of the projected points
pc.variances <- (principal.components$sdev) ^ 2

# calculate proportion explained
pc.proportion.variances <- pc.variances / sum(pc.variances)

# get components collectively explaining 90% of input variance
number.of.components <-
  length(pc.proportion.variances[cumsum(pc.proportion.variances) < .90]) + 1


## Modelling ----
# create train with selected components and target
data.train.pca <-
  data.frame(class = as.factor(features.train$class),
             principal.components$x[, 1:number.of.components])

# perform PCA for test too and get desired number of components
data.test.pca <-
  data.frame(predict(principal.components, newdata = features.test %>%
                       select(-c("id", "class"))))[, 1:number.of.components]

# perform classification again with select algorithms from set
### Support-Vector Machine ----
# linear kernel
tune.out <- tune(
  svm,
  class ~ .,
  data = data.train.pca,
  kernel = "linear",
  ranges = list(cost = c(0.001, 0.005, 0.007, 0.009, 0.01, 0.12))
)

summary(tune.out)

svm.l.features.pca <- svm(
  class ~ .,
  data = data.train.pca,
  kernel = "linear",
  cost = 0.009,
  scale = FALSE
)

summary(svm.l.features.pca)

svm.l.pred.pca <- predict(svm.l.features.pca, data.test.pca)

svm.l.pred.pca <- mode_aggregate(svm.l.pred.pca, features.test$id)

prop.table(table(predict = svm.l.pred.pca, truth = true.class), margin = 2)

acc.svm.l.pca <-
  (table(svm.l.pred.pca,
         true.class)[1] + table(svm.l.pred.pca,
                                true.class)[4]) / (nrow(data.test.pca) /
                                                     3)

f1score.svm.l.pca <- F1_Score(true.class, svm.l.pred.pca)

# radial kernel
tune.out <- tune(
  svm,
  class ~ .,
  data = data.train.pca,
  kernel = "radial",
  ranges = list(cost = c(0.01, 1, 2, 3, 5),
                gamma = c(0.01, 0.1))
)

summary(tune.out)

svm.r.features.pca <- svm(
  class ~ .,
  data = data.train.pca,
  kernel = "radial",
  cost = 3,
  gamma = 0.01,
  scale = FALSE
)

summary(svm.r.features.pca)

svm.r.pred.pca <- predict(svm.r.features.pca, data.test.pca)

svm.r.pred.pca <- mode_aggregate(svm.r.pred.pca, features.test$id)

prop.table(table(predict = svm.r.pred.pca, truth = true.class), margin = 2)

acc.svm.r.pca <-
  (table(svm.r.pred.pca,
         true.class)[1] + table(svm.r.pred.pca,
                                true.class)[4]) / (nrow(data.test.pca) /
                                                     3)

f1score.svm.r.pca <- F1_Score(true.class, svm.r.pred.pca)

## Comparison Results ----
accuracies.pca <-
  data.frame(
    Kernel = rep(c("Linear",
                   "Radial"), 2),
    Method = rep(c(rep(
      "Feature Selection", 2
    ),
    rep("PCA", 2)), 2),
    Metric = c(rep("Overall Accuracy", 4), rep("F1Score", 4)),
    Value = c(
      acc.svm.l,
      acc.svm.r,
      acc.svm.l.pca,
      acc.svm.r.pca,
      f1score.svm.l,
      f1score.svm.r,
      f1score.svm.l.pca,
      f1score.svm.r.pca
    )
  )

# plot difference in performance
ggplot(accuracies.pca, aes(Kernel, Value, fill = Method, label = Value)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(
    position = position_dodge(width = 1),
    aes(
      y = Value + 0.25,
      label = round(Value, 2),
      hjust = 1.5
    ),
    angle = 90
  ) +
  facet_wrap( ~ Metric)
