library(MASS)

setwd("/Users/jiechang/Desktop/IS507/Groupassignment")
Fifa = read.csv("data.csv",header = TRUE)
View(Fifa)

#Check Dimensions of Dataset
dim(Fifa)

#Check Variable Names
names(Fifa)

#Check Data structure
str(Fifa)

#delete rows containing missing value
Fifa = na.omit(Fifa) 

#Check Missing Data
sum(is.na(Fifa))

Fifa = as.data.frame(Fifa[, c(22,55:88)])
library(plyr)
table(Fifa$Position)

# class1 means "goalkeeping positions", class2 means "defensive positions", class3 means "midfield positions", class4 means "attacking positions"
Fifa$Position <- revalue(Fifa$Position, c("GK"="1", "CB"="2","LB"="2","RB"="2","LWB"="2","RWB"="2","RCB"="2","LCB"="2","CDM"="3","RDM"="3","LDM"="3","CAM"="3","LAM"="3","RAM"="3","LW"="3","CM"="3","LM"="3","RM"="3","RW"="3","RCM"="3","LCM"="3","CF"="4","ST"="4","LF"="4","RF"="4","LS"="4","RS"="4"))
Fifa$Position <- as.numeric(Fifa$Position)

require(caTools)  # loading caTools library
library(caTools)
set.seed(123)   #  set seed to ensure you always have same random numbers generated
sample = sample.split(Fifa,SplitRatio = 0.70) # splits the data in the ratio mentioned in SplitRatio. After splitting marks these rows as logical TRUE and the the remaining are marked as logical FALSE
train =subset(Fifa,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test=subset(Fifa, sample==FALSE)

FifaLDA = lda(Position ~ ., data=train)
FifaLDA


#ggplot
ggplot(cbind(train, predict(FifaLDA)$x), aes(LD1, LD2, color = Position)) +
  geom_point() +
  stat_ellipse(level = 0.95, show.legend = FALSE)

#train
predictions <- predict(FifaLDA, train)
predictions

mean(predictions$class == train$Position)
#test
prd<-predict(FifaLDA , test)
predict<-prd$class
table(predict, test$Position)
initial<-as.factor(test$Position)
predicts<-as.factor(predict)

library(caret)
#Confusion Matrix
confusionMatrix(initial, predicts)







