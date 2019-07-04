rm(list = ls())
gc()

#Reading and manually checking

irisD <- read.csv("irisD.csv")
irisD$Species <- as.character(irisD$Species)
str(irisD)

iris_complete <- na.omit(irisD)
nrow(iris_complete)/nrow(irisD)

#Checking with rules
library(editrules)

E<-editfile("irisR.txt")
ve <- violatedEdits(E, irisD)
summary(ve)
plot(ve)
petalError <- which(ve[,8])
petalError

boxplot(irisD$Sepal.Length)
outliers <- boxplot.stats(irisD$Sepal.Length)$out
i <- irisD$Sepal.Length %in% outliers
irisD$Sepal.Length[i] <- rep(NA, length(i))
boxplot(irisD$Sepal.Length)

#Exercice 3: Correcting
le <- localizeErrors(E, irisD, method = "mip")
for(i in 1:ncol(le$adapt)){
  irisD[le$adapt[[i]], i] <- NA
}

#Exercice 4: Imputing
library(VIM)
irisF1 <- kNN(irisD)
ve <- violatedEdits(E, irisF1)
summary(ve)

irisF3 <- irisD
irisF3 <- hotdeck(irisD, variable = "Petal.Width" ,ord_var="Species" )
ve <- violatedEdits(E, irisF3)
summary(ve)
