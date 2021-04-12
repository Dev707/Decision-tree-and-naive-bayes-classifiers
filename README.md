# Decision-tree-and-naive-bayes-classifiers
Decision Tree and Naïve Bayes classifiers

## Contents

- Decision Tree and Naïve Bayes classifiers
- Data preprocessing
- Classification
- ROC curve
- KNN
- Conclusion

## Decision Tree and Naïve Bayes classifiers

This is the Dataset of diabetes, downloaded from the website of the kaggle, taken from the hospital 
Frankfurt, Germany. The datasets consist of several medical predictor (independent) variables and one 
target (dependent) variable, Outcome. Independent variables include the number of pregnancies the 
patient has had, their BMI, insulin level, age, and so on.
There are total 768 observations and with nine columns. Pregnancies, Glucose, BloodPressure, 
SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome.The dataframe of the data set 
is shown below
```
> str(diab)
'data.frame': 768 obs. of 9 variables:
$ Pregnancies : int 6 1 8 1 0 5 3 10 2 8 ...
$ Glucose : int 148 85 183 89 137 116 78 115 197 125 ...
$ BloodPressure : int 72 66 64 66 40 74 50 68 70 96 ...
$ SkinThickness : int 35 29 26 23 35 24 32 32 45 32 ...
$ Insulin : int 160 116 175 94 168 112 88 210 543 402 ...
$ BMI : num 33.6 26.6 23.3 28.1 43.1 25.6 31 35.3 30.5 29.8 ...
$ DiabetesPedigreeFunction: num 0.627 0.351 0.672 0.167 2.288 ...
$ Age : int 50 31 32 21 33 30 26 29 53 54 ...
$ Outcome : Factor w/ 2 levels "No","Yes": 2 1 2 1 2 1 2 1 2 2 ..
```

## Data preprocessing

Before we study the data set let’s convert the output variable (‘Outcome’) into a categorical variable. This 
is necessary because our output will be in the form of 2 classes, True or False. Where true, will denote 
that a patient has diabetes, and false denotes that a person is diabetes free. First of all we transform the 
outcomes of the diabetes data set into Yes or No from 1 and 0 to describe as the patient have diabetes or 
not.
While analyzing the structure of the data set, we can see that the minimum values for Glucose, 
Bloodpressure, Skinthickness, Insulin, and BMI are all zero. This is not ideal since no one can have a value 
of zero for Glucose, blood pressure, etc.

## Classification

### Data partitioning

This stage begins with a process called Data Splicing, wherein the data set is split into two parts:
Training set: This part of the data set is used to build and train the Machine Learning model.
Testing set: This part of the data set is used to evaluate the efficiency of the model.

```
prop.table(table(diab$Outcome))*100
No Yes 
65.10417 34.89583
```

For comparing the outcome of the training and testing phase let’s create separate variables that store the 
value of the response variable:
create objects x which holds the predictor variables and y which holds the response variables.

### Naïve Bayes

Now it’s time to load the e1071 package that holds the Naive Bayes function. This is an in-built 
function provided by R.
After loading the package, the below code snippet will create Naive Bayes model by using the 
training data set:
We thus created a predictive model by using the Naive Bayes Classifier.
To check the efficiency of the model, we are now going to run the testing data set on the model, 
after which we will evaluate the accuracy of the model by using a Confusion matrix.

```
Confusion Matrix and Statistics

          Reference
Prediction  No Yes
       No  130  35
       Yes  20  45
                                          
               Accuracy : 0.7609          
                 95% CI : (0.7004, 0.8145)
    No Information Rate : 0.6522          
    P-Value [Acc > NIR] : 0.000245        
                                          
                  Kappa : 0.4488          
                                          
 Mcnemar's Test P-Value : 0.059058        
                                          
            Sensitivity : 0.8667          
            Specificity : 0.5625          
         Pos Pred Value : 0.7879          
         Neg Pred Value : 0.6923          
             Prevalence : 0.6522          
         Detection Rate : 0.5652          
   Detection Prevalence : 0.7174          
      Balanced Accuracy : 0.7146          
                                          
       'Positive' Class : No              
                                          
```

The final output shows that we built a Naive Bayes classifier that can predict whether a person is 
diabetic or not, with an accuracy of approximately 76%.

### Decision Tree

Decision Trees are versatile Machine Learning algorithm that can perform both classification and 
regression tasks. They are very powerful algorithms, capable of fitting complex datasets. Besides, 
decision trees are fundamental components of random forests, which are among the most potent 
Machine Learning algorithms available today.
First step is to distribute the data set into training and the testing with ration of 70 and 30.

Below are the rules which are derived from the decision tree.

```
Rule number: 59 [Outcome=Yes cover=13 (2%) prob=1.00]
 Glucose>=130.5
 BMI>=29.6
 Glucose< 154.5
 Glucose< 152.5
 BMI>=41.55

```
```
Rule number: 117 [Outcome=Yes cover=12 (2%) prob=0.92]
 Glucose>=130.5
 BMI>=29.6
 Glucose< 154.5
 Glucose< 152.5 BMI< 41.55
 Age>=44.5

```

```
Rule number: 15 [Outcome=Yes cover=71 (13%) prob=0.86]
 Glucose>=130.5
 BMI>=29.6
 Glucose>=154.5

```

```
Rule number: 47 [Outcome=Yes cover=22 (4%) prob=0.82]
 Glucose< 130.5
 Age>=28.5
 BMI>=26.35
 DiabetesPedigreeFunction>=0.6375
 Pregnancies>=2.5
```

```
Rule number: 367 [Outcome=Yes cover=28 (5%) prob=0.64]
 Glucose< 130.5
 Age>=28.5
 BMI>=26.35
 DiabetesPedigreeFunction< 0.6375
 Glucose>=93.5
 SkinThickness< 27.5
 BloodPressure< 83
 DiabetesPedigreeFunction< 0.4255
```

```
Rule number: 367 [Outcome=Yes cover=28 (5%) prob=0.64]
 Glucose< 130.5
 Age>=28.5
 BMI>=26.35
 DiabetesPedigreeFunction< 0.6375
 Glucose>=93.5
 SkinThickness< 27.5
 BloodPressure< 83
 DiabetesPedigreeFunction< 0.4255

```

```
Rule number: 46 [Outcome=No cover=13 (2%) prob=0.38]
 Glucose< 130.5
 Age>=28.5
 BMI>=26.35
 DiabetesPedigreeFunction>=0.6375
 Pregnancies< 2.5
```

```
Rule number: 6 [Outcome=No cover=40 (7%) prob=0.35]
 Glucose>=130.5
 BMI< 29.6
```

```
Rule number: 366 [Outcome=No cover=8 (1%) prob=0.25]
 Glucose< 130.5
 Age>=28.5 BMI>=26.35
 DiabetesPedigreeFunction< 0.6375
 Glucose>=93.5
 SkinThickness< 27.5
 BloodPressure< 83
 DiabetesPedigreeFunction>=0.4255

```

```
Rule number: 90 [Outcome=No cover=32 (6%) prob=0.25]
 Glucose< 130.5
 Age>=28.5
 BMI>=26.35
 DiabetesPedigreeFunction< 0.6375
 Glucose>=93.5
 SkinThickness>=27.5

```

```
Rule number: 28 [Outcome=No cover=7 (1%) prob=0.14]
 Glucose>=130.5
 BMI>=29.6
 Glucose< 154.5
 Glucose>=152.5
```

```
Rule number: 28 [Outcome=No cover=7 (1%) prob=0.14]
 Glucose>=130.5
 BMI>=29.6
 Glucose< 154.5
 Glucose>=152.5
```

```
Rule number: 232 [Outcome=No cover=9 (2%) prob=0.11]
 Glucose>=130.5
 BMI>=29.6
 Glucose< 154.5
 Glucose< 152.5
 BMI< 41.55
 Age< 44.5
 BloodPressure>=81
```

```
Rule number: 4 [Outcome=No cover=196 (36%) prob=0.10]
 Glucose< 130.5
 Age< 28.5

```

```
Rule number: 44 [Outcome=No cover=24 (4%) prob=0.04]
 Glucose< 130.5
 Age>=28.5
 BMI>=26.35
 DiabetesPedigreeFunction< 0.6375 Glucose< 93.5

```

```
Rule number: 10 [Outcome=No cover=31 (6%) prob=0.03]
 Glucose< 130.5
 Age>=28.5
 BMI< 26.35

```

```
Rule number: 15 [Outcome=Yes cover=71 (13%) prob=0.86]
 Glucose>=130.5
 BMI>=29.6
 Glucose>=154.5
```

Rule #117: Has the largest cover of 13%. So if Glucose>=130.5, the person is not going to be 
diagnosed with diabetes with a probability of 0.86. Here are conditions below. 

```
Confusion Matrix and Statistics

          Reference
Prediction  No Yes
       No  121  28
       Yes  29  52
                                          
               Accuracy : 0.7522          
                 95% CI : (0.6912, 0.8066)
    No Information Rate : 0.6522          
    P-Value [Acc > NIR] : 0.0007075       
                                          
                  Kappa : 0.4553          
                                          
 Mcnemar's Test P-Value : 1.0000000       
                                          
            Sensitivity : 0.6500          
            Specificity : 0.8067          
         Pos Pred Value : 0.6420          
         Neg Pred Value : 0.8121          
             Prevalence : 0.3478          
         Detection Rate : 0.2261          
   Detection Prevalence : 0.3522          
      Balanced Accuracy : 0.7283          
                                          
       'Positive' Class : Yes             
                                          
```

Accuracy of the model is 0.7739, which seems a fair measure, however, p-value is a low number, 
and this means that the model is performing very well. Sensitivity of 0.625 indicated that only 
62.5% of people with diabetes were predicted to be sick, and specificity of 0.85 indicates that 
85% of people who were not sick, were correctly predicted to be so.

## ROC curve
```
Area under the curve is only 0.811, which is not much more than 0.5 of no information rate
```

## KNN

From the table it is obvious that specificity, sensitivity, ppv and npv are low, so the model is not 
performing very well.

```
 [1] Yes Yes No  Yes Yes No  No  No  Yes No  Yes No  No  No  No  No  No  No  No  No 
Levels: No Yes
     
knn    No Yes
  No  115  38
  Yes  35  42
```

## Conclusion

Neither Decision tree, nor KNN are performing properly. So it is better not to use any of 
those models. We have used the Naive Bayes which performed very well
Project Features:

