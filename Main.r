---
title: "Untitled"
output:
  pdf_document: default
  word_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Diabetes

This is the Dataset of diabetes, downloaded from the website of the kaggle, taken from the hospital Frankfurt, Germany.
There are total 768 observations and with nine columns. Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome.The dataframe of the data set is shown below

```{r, echo=FALSE}
#Loading required packages
library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(rpart)
library(randomForest)
library(rpart.plot)
library(rattle)
library(ROCR)
library(class)

setwd("C:/Users/lodi2/Desktop")
diab<-read.csv("diabetes.csv")
str(diab)
```

##	Data preprocessing 
Before we study the data set let’s convert the output variable (‘Outcome’) into a categorical variable. This is necessary because our output will be in the form of 2 classes, True or False. Where true will denote that a patient has diabetes and false denotes that a person is diabetes free. Fisrt of all we transform the outcomes of the diabetes data set into Yes or No from 1 and 0 to describe as the patient have diabetesor not. 
```{r, echo=FALSE}
diab$Outcome<-factor(diab$Outcome, levels=c(0,1), labels=c("No", "Yes"))
```
While analyzing the structure of the data set, we can see that the minimum values for Glucose, Bloodpressure, Skinthickness, Insulin, and BMI are all zero. This is not ideal since no one can have a value of zero for Glucose, blood pressure, etc. Therefore,  such values are treated as missing observations.

In the below code snippet, we’re setting the zero values to NA’s:

```{r, echo=FALSE}
#Convert '0' values into NA
diab[, 2:7][diab[, 2:7] == 0] <- NA
```

To check how many missing values we have now, let’s visualize the data:
```{r, echo=FALSE}
#visualize the missing data
missmap(diab)
```
The above illustrations show that our data set has plenty missing values and removing all of them will leave us with an even smaller data set, therefore, we can perform imputations by using the mice package in R.

```{r, echo=FALSE}
#Use mice package to predict missing values
mice_mod <- mice(diab[, c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")], method='rf')
mice_complete <- complete(mice_mod)
#Transfer the predicted missing values into the main data set
diab$Glucose <- mice_complete$Glucose
diab$BloodPressure <- mice_complete$BloodPressure
diab$SkinThickness <- mice_complete$SkinThickness
diab$Insulin<- mice_complete$Insulin
diab$BMI <- mice_complete$BMI
```
To check if there are still any missing values, let’s use the missmap plot:

```{r, echo=FALSE}
missmap(diab)
```
The output looks good, there is no missing data.

Now let’s perform a couple of visualizations to take a better look at each variable, this stage is essential to understand the significance of each predictor variable.
```{r, echo=FALSE}
#Data Visualization
#Visual 1
ggplot(diab, aes(Age, colour = Outcome)) +
geom_freqpoly(binwidth = 1) + labs(title="Age Distribution by Outcome")

#visual 2
c <- ggplot(diab, aes(x=Pregnancies, fill=Outcome, color=Outcome)) +
geom_histogram(binwidth = 1) + labs(title="Pregnancy Distribution by Outcome")
c + theme_bw()

#visual 3
P <- ggplot(diab, aes(x=BMI, fill=Outcome, color=Outcome)) +
geom_histogram(binwidth = 1) + labs(title="BMI Distribution by Outcome")
P + theme_bw()

#visual 4
ggplot(diab, aes(Glucose, colour = Outcome)) +
geom_freqpoly(binwidth = 1) + labs(title="Glucose Distribution by Outcome")
```
There is boxplot for the Outcome of diagnosis and number of pregnancies and Correlation of number of pregnancies and BMI.
```{r, echo=FALSE}
ggplot(diab, aes(x=Outcome, y=Pregnancies))+geom_boxplot()+labs(x="Diagnosed with diabetes", y="Number of pregnancies", title="Outcome of diagnosis and number of pregnancies")
```
obvious that the more are the numbers of pregnancies the higher is the positive diagnosis rate. It is also worth to note that the range of number pregnancies is quite big for those diagnosed with diabetes, compared to those who were not diagnosed.

```{r, echo=FALSE}
ggplot(diab, aes(x=Outcome, y=Pregnancies))+geom_boxplot()+labs(x="Diagnosed with diabetes", y="Number of pregnancies", title="Outcome of diagnosis and number of pregnancies")
```
There is an obvious correlation of high BMI and positive diagnosis of diabetes.

## Classification

## Data partitioning 

This stage begins with a process called Data Splicing, wherein the data set is split into two parts:

Training set: This part of the data set is used to build and train the Machine Learning model.
Testing set: This part of the data set is used to evaluate the efficiency of the model.

```{r, echo=FALSE}
#Building a model
#split data into training and test data sets
indxTrain <- createDataPartition(y = diab$Outcome,p = 0.70,list = FALSE)
training <- diab[indxTrain,]
testing <- diab[-indxTrain,] #Check dimensions of the split
prop.table(table(diab$Outcome))*100
```
For comparing the outcome of the training and testing phase let’s create separate variables that store the value of the response variable:

```{r, echo=FALSE}
#create objects x which holds the predictor variables and y which holds the response variables
x = training[,-9]
y = training$Outcome
```
## Naïve Bayes
Now it’s time to load the e1071 package that holds the Naive Bayes function. This is an in-built function provided by R.

```{r, echo=FALSE}
library(e1071)
```
After loading the package, the below code snippet will create Naive Bayes model by using the training data set:

```{r, echo=FALSE}
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
model
```
We thus created a predictive model by using the Naive Bayes Classifier.

To check the efficiency of the model, we are now going to run the testing data set on the model, after which we will evaluate the accuracy of the model by using a Confusion matrix.

```{r, echo=FALSE}
Predict <- predict(model,newdata = testing ) #Get the confusion matrix to see accuracy value and other parameter values 
confusionMatrix(Predict, testing$Outcome )
```
The final output shows that we built a Naive Bayes classifier that can predict whether a person is diabetic or not, with an accuracy of approximately 76%.

To summaries the demo, let’s draw a plot that shows how each predictor variable is independently responsible for predicting the outcome.

```{r, echo=FALSE}
#Plot Variable performance
X <- varImp(model)
plot(X)
```
From the above illustration, it is clear that ‘Glucose’ is the most significant variable for predicting the outcome.

## Decision Tree

```{r, echo=FALSE}
set.seed(50)
index<-createDataPartition(diab$Outcome, p=0.7, list=FALSE)
Train<-diab[index,]
Test<-diab[-index,]
```



## Decision tree

```{r, echo=FALSE}
set.seed(50)
model_dt<-rpart(Outcome~., data=Train)
rpart.plot(model_dt)
asRules(model_dt)
```

Rule #4 has the largest cover of 35%. So if Glucose<127.5 and age is less than 28.5 the person is not going to be diagnosed with diabetes with a probability of 0.92.

```{r, echo=FALSE}
pred1<-predict(model_dt, Test, type="class" )
confusionMatrix(pred1, reference = Test$Outcome, positive="Yes")
```
Accuraccy of the model is 0.7739, which seems a fair measure, however, p-value is a low number, and this means that the model is  performing very well. Sensitivity of 0.625 indicated that only 62.5% of people with diabetes were predicted to be sick, and specificity of 0.85 indicates that 85% of people who were not sick, were correctly predicted to be so.

ROC curve
```{r, echo=FALSE}
pred1_prob<-predict(model_dt, Test)
prediction1<-prediction(pred1_prob[,2],Test$Outcome )
perf1<-performance(prediction1, "tpr", "fpr")
plot(perf1, colorize=TRUE)
```

```{r, echo=FALSE}
performance(prediction1, "auc")@y.values
```

Area under the curve is only 0.811, which is not much more than 0.5 of no information rate.

## KNN

```{r, echo=FALSE}
knn<-knn(train = Train[,-9], test=Test[,-9], cl=Train$Outcome, k=5)
knn[1:20]

table(knn, Test$Outcome)
```

From the table it is obvious that specificity, sensitivity, ppv and npv are low, so the model is not performing very well.

Conclusion: Neither Decision tree, nor KNN are performing properly. So it is better not to use any of those models. We have used the Naive Bayes which performed very well 

```{r, echo=FALSE}
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model <- train(Fertility ~., data = swiss, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)
```