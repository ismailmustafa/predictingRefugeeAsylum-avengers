require(dplyr)
require(rpart)
require(caret)

#Load Data
data = read.csv("C:/Stuff/NYU/Courses/ML/project/git/PredictingRefugeeAsylum-avengers/data/asylum_data_complete.csv")

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

#Fit the decision tree model
Classification_Model = rpart(grantraw~comp_date+lawyer+natid+written+adj_time_start+
					flag_earlystarttime+numinfamily+numfamsperslot+
					numfamsperday+orderwithinday+L1grant+L1grant_sameday+
					L2grant+numgrant_prev5+numcourtgrant_prev5+numcourtdecideself_prev5+
					numcourtgrantother_prev5+					
					year+natdefcode+courtid+
					samenat+grantgrant+
					grantdeny+denygrant+denydeny+hour_start+morning+
					lunchtime+numcases_judgeday+numcases_judge+
					numcases_court_hearing+Year_Appointed_SLR_x+
					experience+experience8+Gender+Government_Years_SLR+
					Govt_nonINS_SLR+INS_Years_SLR+INS_Every5Years_SLR+
					Military_Years_SLR+NGO_Years_SLR+Privateprac_Years_SLR+
					Academia_Years_SLR+Bar,data = train)

#Get Variable importance
varImp(Classification_Model)

#Get model parameters
plotcp(Model)
printcp(Model)

#Prune the tree to avoid overfitting
Pruned_Model = prune(Model, cp = 0.02)
plot(Pruned_Model, uniform = TRUE)
text(Pruned_Model, use.n = TRUE, cex =0.75)
TrainPredictions = predict(object = Pruned_Model, newdata=train, type = "vector")

TestPredictions = predict(object = Pruned_Model, newdata=test, type = "vector")


#Training and Test Accuracies
cat('Test Accuracy :', 1- sum(abs(grant_test_sports - TestPredictions))/length(TestPredictions))
cat('Train Accuracy :', 1- sum(abs(grant_sports - TrainPredictions))/length(TrainPredictions)))