require(dplyr)
require(gbm)

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

#Load Data
data_sports = read.csv(args[1])
print('Data Loaded')

## 75% of the sample size
smp_size_sports <- floor(0.75 * nrow(data_sports))

## set the seed to make your partition reproductible
set.seed(123)
train_ind_sports <- sample(seq_len(nrow(data_sports)), size = smp_size_sports)

#Split data to train and test
train_sports <- data_sports[train_ind_sports, ]
test_sports <- data_sports[-train_ind_sports, ]

#Get the target variable
grant_sports = train_sports$grantraw
grant_test_sports = test_sports$grantraw
train_sports = select(train_sports, -grantraw)
test_sports = select(test_sports, -grantraw)
end_trn = nrow(train_sports)

#Combine test and train data to apply changes easily
all_sports = rbind(train_sports, test_sports)
end = nrow(all_sports)

#Select relevant features
all_selected_sports = select(all_sports,
								Bar,
								lawyer,
								natdefcode,
								natid,
								numcases_court_hearing,
								numcourtgrant_prev5,
								numcourtgrantother_prev5,
								numgrant_prev5,
								comp_date,
								written,
								adj_time_start,
								flag_earlystarttime,
								numinfamily,
								numfamsperslot,
								numfamsperday,
								orderwithinday,
								L1grant,
								L1grant_sameday,
								L2grant,
								numcourtdecideself_prev5,
								year,
								courtid,
								samenat,
								grantgrant,
								grantdeny,
								denygrant,
								denydeny,
								hour_start,
								morning,
								lunchtime,
								numcases_judgeday,
								numcases_judge,
								Year_Appointed_SLR_x,
								experience,
								experience8,
								Gender,
								Government_Years_SLR,
								Govt_nonINS_SLR,
								INS_Years_SLR,
								INS_Every5Years_SLR,
								Military_Years_SLR,
								NGO_Years_SLR,
								Privateprac_Years_SLR,
								Academia_Years_SLR,
								prcp,
								snow, 
								snwd, 
								tmax, 
								tmin, 
								tsun,
								prcp_minus_1, 
								snow_minus_1, 
								snwd_minus_1, 
								tmax_minus_1, 
								tmin_minus_1, 
								tsun_minus_1,
								prcp_minus_2, 
								snow_minus_2, 
								snwd_minus_2, 
								tmax_minus_2, 
								tmin_minus_2, 
								tsun_minus_2,
								prcp_minus_3, 
								snow_minus_3, 
								snwd_minus_3, 
								tmax_minus_3, 
								tmin_minus_3, 
								tsun_minus_3,
								prcp_minus_4, 
								snow_minus_4, 
								snwd_minus_4, 
								tmax_minus_4, 
								tmin_minus_4, 
								tsun_minus_4,
								nba_undergrad,
								nba_lawschool,
								nba_bar,
								nfl_undergrad,
								nfl_lawschool, 
								nfl_bar,
								mlb_undergrad,
								mlb_lawschool, 
								mlb_bar,
								nhl_undergrad,
								nhl_lawschool,
								nhl_bar
								)


#Number of steps tuning		
depthVec = vector()
trainAcc = vector()
testAcc = vector()
i=0
for (depth in c(1,10, 100, 1000, 2000, 5000, 10000, 20000, 25000, 30000)){
	i=i+1	
	ntrees = depth
	depthVec[i] = depth

	Model = gbm.fit(x=all_selected[1:end_trn,]
			,y = grant
			,distribution = "adaboost"
			,n.trees = ntrees
			,shrinkage = 0.01
			, interaction.depth = 2
			, n.minobsinnode = 1000
			, nTrain = round(end_trn * 0.8)
			, verbose = FALSE
		)


	
	TrainPredictions = predict(object = Model, newdata=all_selected[1:end_trn,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")

	TestPredictions = predict(object = Model, newdata=all_selected[(end_trn+1):end,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")
	TestPredictions  = round(TestPredictions)
	TrainPredictions = round(TrainPredictions)

	testAcc[i]= 1- sum(abs(grant_test - TestPredictions))/length(TestPredictions)
	trainAcc[i] = 1- sum(abs(grant - TrainPredictions))/length(TrainPredictions)  
}

g_range <- range(0, testAcc, trainAcc)

plot(depthVec, testAcc, type="o", col="blue", ann=FALSE)
lines(depthVec, trainAcc, type="o", pch=22, lty=2, col="red", ann=FALSE)
# Create a title with a red, bold/italic font
title(main="Number of Steps Vs Accuracy", col.main="blue", font.main=4)

# Label the x and y axes with dark green text
title(xlab="Number of Steps", col.lab=rgb(0,0.5,0))
title(ylab="Accuracy", col.lab=rgb(0,0.5,0))

# Create legend
legend(1, g_range[2], c("Test","Train"), cex=0.8, 
   col=c("blue","red"), pch=21:22, lty=1:2);

#Tuning number of nodes in the leaf nodes
nodeVec = vector()
trainNodeAcc = vector()
testNodeAcc = vector()
i=0
for (node in c(1,10, 100, 1000, 2000, 4000,4500,5000)){
	i=i+1	
	nodeVec[i] = node

	Model = gbm.fit(x=all_selected[1:end_trn,]
			,y = grant
			,distribution = "adaboost"
			,n.trees = 20000
			,shrinkage = 0.1
			, interaction.depth = 2
			, n.minobsinnode = node
			, nTrain = round(end_trn * 0.8)
			, verbose = FALSE
		)


	TrainPredictions = predict(object = Model, newdata=all_selected[1:end_trn,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")

	TestPredictions = predict(object = Model, newdata=all_selected[(end_trn+1):end,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")
	TestPredictions  = round(TestPredictions)
	TrainPredictions = round(TrainPredictions)
	
	testNodeAcc[i]= 1- sum(abs(grant_test - TestPredictions))/length(TestPredictions)
	trainNodeAcc[i] = 1- sum(abs(grant - TrainPredictions))/length(TrainPredictions)  
}

g_range <- range(testNodeAcc, trainNodeAcc)
plot(nodeVec, testNodeAcc, ylim=g_range, type="o", col="blue", ann=FALSE)
lines(nodeVec, trainNodeAcc, type="o",pch=22, lty=2, col="red", ann=FALSE)
abline(v=1000, col="green", lwd=3, lty=2)
# Create a title with a red, bold/italic font
title(main="Complexity(# of observations in Leaf Nodes) Vs Accuracy", col.main="blue", font.main=4)

# Label the x and y axes with dark green text
title(xlab="Minimum number of observations in the Trees Leaf nodes", col.lab=rgb(0,0.5,0))
title(ylab="Accuracy", col.lab=rgb(0,0.5,0))

# Create a legend at (1, g_range[2]) that is slightly smaller 
# (cex) and uses the same line colors and points used by 
# the actual plots 
legend("topright", g_range[2], c("Test","Train"), cex=0.8, 
   col=c("blue","red"), pch=21:22, lty=1:4);


ShrinkageVec = vector()
trainShrinkageAcc = vector()
testShrinkageAcc = vector()
i=0
for (shrink in c(1, 0.1, 0.01, 0.001, 0.0001)){
	i=i+1	
	ShrinkageVec [i] = shrink

	Model = gbm.fit(x=all_selected[1:end_trn,]
			,y = grant
			,distribution = "adaboost"
			,n.trees = 25000
			,shrinkage = shrink
			, interaction.depth = 2
			, n.minobsinnode = 1000
			, nTrain = round(end_trn * 0.8)
			, verbose = FALSE
		)

	TrainPredictions = predict(object = Model, newdata=all_selected[1:end_trn,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")

	TestPredictions = predict(object = Model, newdata=all_selected[(end_trn+1):end,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")
	TestPredictions  = round(TestPredictions)
	TrainPredictions = round(TrainPredictions)

	testShrinkageAcc [i]= 1- sum(abs(grant_test - TestPredictions))/length(TestPredictions)
	trainShrinkageAcc [i] = 1- sum(abs(grant - TrainPredictions))/length(TrainPredictions)  
}

g_range <- range(testShrinkageAcc, trainShrinkageAcc)
plot(ShrinkageVec, testShrinkageAcc, ylim=g_range, type="o", col="blue", ann=FALSE)
lines(ShrinkageVec, trainShrinkageAcc, type="o",pch=22, lty=2, col="red", ann=FALSE)
abline(v=0.1, col="green", lwd=3, lty=2)
# Create a title with a red, bold/italic font
title(main="Shrinkage Factor Vs Accuracy", col.main="blue", font.main=4)

# Label the x and y axes with dark green text
title(xlab="Shrinkage Factor", col.lab=rgb(0,0.5,0))
title(ylab="Accuracy", col.lab=rgb(0,0.5,0))

# Create a legend at (1, g_range[2]) that is slightly smaller 
# (cex) and uses the same line colors and points used by 
# the actual plots 
legend("topright", g_range[2], c("Test","Train"), cex=0.8, 
   col=c("blue","red"), pch=21:22, lty=1:4)