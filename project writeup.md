---
title: "Project Writeup"
author: "Alex Tilcock"
output: html_document:
        pandoc_args: [
      "+RTS", "-K64m",
      "-RTS"
    ]
---
# Background 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

Six participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Accelerometers on the belt, forearm, arm, and barbell recorded the activity. Each activity was assigned a class where Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes

More information about the data is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)

# Assignment Goal 

The goal of this assignment is use the captured data to create a machine learning model that is able to predict the class that should be assigned based on the values captured by the accelerometers. 

The assignment provides two data files named pml-training.csv and pml-testing.csv.  I have elected to use the pml-training file to perform cross-validation, splitting the data into a training and test set.  The pml-testing.csv file will be used for final validation.

# Explore and clean-up training data 

Intial evaluation of the pml-training data reveals that there are a number of issues that need to be addressed before the data can be reliably used to create a model.  To keep the write up within the required 2000 word/5 figure limit I have commented out the lines that describe and evaluate the data but these can be executed by removing the comments marks if desired.  The code simply loads the data, removes the #DIV/0! entries and converts the non-numeric values to numeric; 



```r
## load required libraries 
if (!require("Hmisc")) install_packages("Hmisc")
library(Hmisc)
if (!require("caret")) install_packages("caret")
library(caret)

# load training dataset
training<-read.csv("pml-training.csv")

## review data characteristics - uncomment these two lines to see the result 
##describe(training)
##nearZeroVar(training,saveMetrics=TRUE)


## clean up Divide by zero 
training[training=="#DIV/0!"]<-NA

ncols<-ncol(training)

## ensure all numeric columns (from 8 to end of data set except the last column (Classe) are numeric 
training[,c(8:(ncols-1))]<-sapply(training[,c(8:(ncols-1))],as.numeric)
```

# Select Features

Now that the data has been cleaned up I want to do some reduction to limit the predictors to those that are relevant. This code addresses the following issues;

1. Many vectors contain a high percentage of NA values.
2. Some of the  vectors are not relevant (i.e. index column, timestamps)
3. A large number of vectors have near zero variance 



```r
## remove first seven columns as they are not relevant features for the prediction
training<-training[,-c(1:7)]

## identify any columns that have near zero variance   
nsv <- nearZeroVar(training,saveMetrics=TRUE)

## remove columns with no significant variance 
nsvCols <- rownames(nsv[nsv$nzv==TRUE,])
training<-training[,-which(names(training) %in% c(nsvCols))]

## identify columns that have NA's present 

nalist <- data.frame(colSums(is.na(training)))
naCols <- rownames(subset(nalist,nalist>0))

## remove NA columns to arrive at the final dataset for training the model
trainingFinal<-training[,-which(names(training) %in% c(naCols))]
```


# Create datasets for cross validation 

After running the clean up I now have a training data set consisting of 19622 observeration and 53 variables. I will partition this training set into two sets for training and testing of the model. 


```r
inTrain<-createDataPartition(trainingFinal$classe,p=0.75,list=FALSE)
trainPart <- trainingFinal[inTrain,]
testPart <- trainingFinal[-inTrain,]
```

# Run two different models  

Now I will use these two partitions to create and validate two separate models, one using random forest and a second using a CART decision tree.  ; 


```r
set.seed(1999)
modelRF <- train(classe ~ . ,data=trainPart, method="rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```

```r
predRF <- predict(modelRF,testPart)
```



```r
set.seed(1999)
modelRpart <- train(classe ~ . ,data=trainPart, method="rpart")
```

```
## Loading required package: rpart
```

```r
predRpart <- predict(modelRpart,testPart)
```

# Evaluate error rate and performance for two different models  

Evaluating the in sample and out of sample accuracy for the random forest model



```r
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.67%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4184    1    0    0    0 0.0002389486
## B   17 2825    6    0    0 0.0080758427
## C    0   18 2546    3    0 0.0081807557
## D    0    0   42 2368    2 0.0182421227
## E    0    0    3    7 2696 0.0036954915
```
                      
                       Figure 1 - Random Forest Model


```r
confusionMatrix(predRF,testPart$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1391    2    0    0    0
##          B    3  947    4    0    0
##          C    0    0  849    7    0
##          D    0    0    2  797    2
##          E    1    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9935, 0.9973)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9971   0.9979   0.9930   0.9913   0.9978
## Specificity            0.9994   0.9982   0.9983   0.9990   0.9998
## Pos Pred Value         0.9986   0.9927   0.9918   0.9950   0.9989
## Neg Pred Value         0.9989   0.9995   0.9985   0.9983   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2836   0.1931   0.1731   0.1625   0.1833
## Detection Prevalence   0.2841   0.1945   0.1746   0.1633   0.1835
## Balanced Accuracy      0.9983   0.9981   0.9956   0.9952   0.9988
```

                    Figure 2 - Random Forest Prediction



```r
modelRpart[4]
```

```
## $results
##           cp  Accuracy      Kappa AccuracySD    KappaSD
## 1 0.03484287 0.5102480 0.36070927 0.01440269 0.02035171
## 2 0.06104624 0.4286487 0.22932351 0.06534795 0.10889660
## 3 0.11639609 0.3372564 0.08069351 0.04208114 0.06182158
```
                    Figure 3 - Decision Tree Model 
                   


```r
confusionMatrix(predRpart,testPart$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1261  392  407  355  127
##          B   24  319   35  149  131
##          C  107  238  413  300  249
##          D    0    0    0    0    0
##          E    3    0    0    0  394
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4867          
##                  95% CI : (0.4727, 0.5008)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3293          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9039  0.33614  0.48304   0.0000  0.43729
## Specificity            0.6349  0.91429  0.77920   1.0000  0.99925
## Pos Pred Value         0.4961  0.48480  0.31599      NaN  0.99244
## Neg Pred Value         0.9433  0.85163  0.87712   0.8361  0.88751
## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
## Detection Rate         0.2571  0.06505  0.08422   0.0000  0.08034
## Detection Prevalence   0.5184  0.13418  0.26652   0.0000  0.08095
## Balanced Accuracy      0.7694  0.62521  0.63112   0.5000  0.71827
```
              
              Figure 4 - Decision Tree Model prediction

# Summary

Examining the expected error rate and 95% CI values for the Random Forest model vs Decision tree shows significant discrepencies.  The RF model error rate is less than 1% while the error rate for the Decision Tree model is close to 50%. This is also reflected in the outputs of the prediction using the same test data against each model. It is apparent that the random forest model provided a much higher level of accuracy against the training data set than the decision tree.  I will validate this using the pml-testing data.


```r
## create function to write out results to separate files 
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

# load testing dataset
validation<-read.csv("pml-testing.csv")

predValRF<-predict(modelRF,validation)

predValRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files(predValRF)
```


# Conclusion

Submission of the output of the validation data set against the Random Forest model confirmed 20/20 correct predictions.  This would appear to confirm that the model is reliable. An assessment of the important variables to the model (using r varImp function) shows a significant reduction in importance in the first 20 variables.  Although not required for this assignment it is possible to use this inforemation to perform a further reduction in the number of predictors which would speed up the creation of the model. 
                   
