rm(list=ls())
#stwd('C:\Users\win7\Documents\R files')

data("iris")

str(iris)
summary(iris)
table(iris$Species)
irisClass=iris$Species
irisData=iris[-5]
#dim(irisData)
normalize= function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
irisData_norm= as.data.frame(lapply(irisData,normalize))
#View(irisData_norm)
summary(irisData_norm)

kc=kmeans(irisData_norm,3)
table(irisClass,kc$cluster)
plot(irisData_norm[c("Sepal.Length","Sepal.Width")],col=kc$cluster)
points(kc$centers[,c("Sepal.Length","Sepal.Width")],col=1:3,pch=23,cex=3)

