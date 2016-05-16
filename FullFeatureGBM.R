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

#Fit the model
Model = gbm.fit(x=all_selected_sports[1:end_trn,]
			,y = grant_sports
			,distribution = "adaboost"
			,n.trees = 20000
			,shrinkage = 0.1
			, interaction.depth = 2
			, n.minobsinnode = 1000
			, nTrain = round(end_trn * 0.8)
			, verbose = FALSE
		)


#Predict Train and Test predictions	
TrainPredictions = predict(object = Model, newdata=all_selected_sports[1:end_trn,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")
TestPredictions = predict(object = Model, newdata=all_selected_sports[(end_trn+1):end,], n.trees = gbm.perf(Model, plot.it = FALSE)
							, type = "response")
TestPredictions  = round(TestPredictions)
TrainPredictions = round(TrainPredictions)

#Training and Test Accuracies
cat('Test Accuracy :', 1- sum(abs(grant_test_sports - TestPredictions))/length(TestPredictions))
cat('Train Accuracy :', 1- sum(abs(grant_sports - TrainPredictions))/length(TrainPredictions)))
cat('Area under ROC curve :', gbm.roc.area(grant_test_sports,TestPredictions))

