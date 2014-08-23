Practical Machine Learning Course Project
========================================================
## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here:  [http://groupware.les.inf.puc-rio.br/har] . 

## Data 


The training data for this project are available here: 

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]

The test data are available here: 

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv]

## Analysis
### Goal
The goal of this analysis is to predict the manner in which they did the exercise. Therefore we need to find the best model to fit the training data, which will be furthur used to predict on the test dataset.

### Preprocessing

As always, the first job is to download and read in the data. Also package *caret* will be needed for this anlysis. 



```r
training<-read.csv("pml-training.csv")
testing<-read.csv("pml-testing.csv")
```

The training dataset has 19622 observations and 160 variable, which contains the outcome *classe* and 159 predictors. 

```r
dim(training)
```

```
## [1] 19622   160
```

The testing dataset has 20 observation and do not have variable *classe*, which we need to predict.

While looking at the 159 predictors in training dataset, we don't necessarily need every one of them. Hence I tried to removed the NA columns and variables that are obviously irrelevent. 

```r
training <- training[, which(as.numeric(colSums(is.na(training)))==0)]
training <- training[, which(as.numeric(colSums(training==""))==0)]
training<-training[,c(6:60)]
training$new_window<-I(as.character(training$new_window)=="yes")*1
```
I ended up with 55 variables. Although there is certainly space for improvement. The more preprocess we do, the less time it will takes to calculate the model. (I am pretty lazy so I decided to keep R running over night)

### Cross validation

For the purpose of cross validation, we will need to split the *training* dataset into *train* and *test*, while the so called *testing* dataset is not actually used in this part. I take 70% of the sample data into *train* and 30% into *test*



```r
intrain<-createDataPartition(y=training$classe,p=0.7,list=FALSE)
train<-training[intrain,]
test<-training[-intrain,]
```

Here is how the classe variable looks like in *train*.


```r
ggplot(data=train,aes(x=classe,fill=classe))+geom_histogram(binwidth=0.05)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 
### Fit model

Now that  the pre-processing is done, we should go for the most important step, which is fitting the model. While there are a couple of methods we can use, I decided to go with the *random forest method*. This method is known for accuracy. But it usually takes more time. Since the dataset we have is not huge, I would prefer to sacrifice time for the model accuracy.

Here is the code to apply the *random forest method*. I also used principla components analysis with 95% of variation captured. 


```r
fit<-train(classe~.,data=train,method="rf",
           preProcess="pca",verbose=FALSE,na.remove=TRUE,thresh=0.95)
```





```r
plot(fit)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9.png) 

The model fitted well on the training set.

```r
fit
```

```
## Random Forest 
## 
## 13737 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction, scaled, centered 
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1.0       1.0    0.004        0.004   
##   28    0.9       0.9    0.008        0.010   
##   54    0.9       0.9    0.008        0.011   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

100% accurate on training set.

```r
confusionMatrix(train$classe,predict(fit,newdata=train))
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
### Accuracy
Let's test how accurate the model is on the *test dataset*.

```r
confusionMatrix(predict(fit,newdata=test),test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1660   13    2    1    0
##          B    5 1109   19    0    7
##          C    4   16  985   37    5
##          D    4    0   20  924    7
##          E    1    1    0    2 1063
## 
## Overall Statistics
##                                         
##                Accuracy : 0.976         
##                  95% CI : (0.971, 0.979)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.969         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.974    0.960    0.959    0.982
## Specificity             0.996    0.993    0.987    0.994    0.999
## Pos Pred Value          0.990    0.973    0.941    0.968    0.996
## Neg Pred Value          0.997    0.994    0.992    0.992    0.996
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.188    0.167    0.157    0.181
## Detection Prevalence    0.285    0.194    0.178    0.162    0.181
## Balanced Accuracy       0.994    0.984    0.974    0.976    0.991
```
As shown above, the model has 97.55% accuracy on the test dataset, which is pretty accurate. The out-of-sample error rate is 2.45%

Here is a plot with accuracy by different classe.

```r
typecnt<-data.frame(table(test$classe))
correct<-data.frame(table(test$classe[prediction==test$classe]))
correct$accuracy=correct$Freq/typecnt$Freq
colnames(correct)[1]<-"classe"
qplot(data=correct,x=classe,y=accuracy,colour=classe)+geom_point(size=5)
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13.png) 

### Predicting
With a good model, it is straight foward to make prediction on *testing dataset*.


```r
data.frame(predict(fit,testing))
```

