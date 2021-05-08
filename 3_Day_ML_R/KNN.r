rm(list=ls())
#stwd('C:\Users\win7\Documents\R files')

data("iris")

str(iris)
summary(iris)
table(iris$Species)
library(ggplot2)
ggplot(iris, aes(Petal.Length,Petal.Width, color= Species)) + geom_point()
ind = sample(2,nrow(iris),replace= TRUE,prob= c(.7,.3))
trainData= iris[ind==1,]
testData= iris[ind==2,]
View(trainData)
trnData= trainData[-5]
tstData= testData[-5]
dim(trnData)
dim(tstData)
normalize= function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
trnData_norm= as.data.frame(lapply(trnData,normalize))
tstData_norm= as.data.frame(lapply(tstData,normalize))
trainLabels=trainData$Species
dim(trainLabels)
testLabels=testData$Species


library(class)
pred_testLabels= knn(train = trnData_norm,test = tstData_norm,cl=trainLabels, k=3,prob = TRUE)
table(testLabels)
table(pred_testLabels)

table(testLabels,pred_testLabels)
acc=mean(testLabels==pred_testLabels)
print(acc)
