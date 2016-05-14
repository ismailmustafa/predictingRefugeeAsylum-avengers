# Predicting Refugee Asylum - Avengers


### Directory structure

![Directory Structure](/images/directoryStructure.png)

### Usage

Make sure your directory is structured as shown in the picture above.

### Clean

run the clean file as follows:

```bash
python clean.py
```

This will look for a file called "asylum_clean.csv" in the data folder and it
will generate a file called "complete_data.csv". The following features are
added in the clean process:

* LastName
* FirstName
* Gender
* FirstUndergrad
* JudgeUndergradLocation
* LawSchool
* JudgeLawSchoolLocation
* Bar
* OtherLocationsMentioned
* IJ_NAME
* Judge_name_SLR
* Male_judge
* Court_SLR
* DateofAppointment
* Year_Appointed_SLR_y
* YearofFirstUndergradGraduatio
* Year_College_SLR
* Year_Law_school_SLR
* President_SLR
* Government_Years_SLR
* Govt_nonINS_SLR
* INS_Years_SLR
* INS_Every5Years_SLR
* Military_Years_SLR
* NGO_Years_SLR
* Privateprac_Years_SLR
* Academia_Years_SLR
* judge_name_caps
* city
* nat_code
* nationality
* twitter_score
* prcp
* snow
* snwd
* tmax
* tmin
* tsun
* prcp_minus_1
* snow_minus_1
* snwd_minus_1
* tmax_minus_1
* tmin_minus_1
* tsun_minus_1
* prcp_minus_2
* snow_minus_2
* snwd_minus_2
* tmax_minus_2
* tmin_minus_2
* tsun_minus_2
* prcp_minus_3
* snow_minus_3
* snwd_minus_3
* tmax_minus_3
* tmin_minus_3
* tsun_minus_3
* prcp_minus_4
* snow_minus_4
* snwd_minus_4
* tmax_minus_4
* tmin_minus_4
* tsun_minus_4
* nba_undergrad
* nba_lawschool
* nba_bar
* nfl_undergrad
* nfl_lawschool
* nfl_bar
* mlb_undergrad
* mlb_lawschool
* mlb_bar
* nhl_undergrad
* nhl_lawschool
* nhl_bar

### HMM

To run the HMM, type the following:

```bash
python hmm.py
```

This will train the hmm using the file you generated in the previous step. It
will also optimize the transition probabilities of the HMM and print out all
accuracies greater than 0.9.

### Adaboost, Decision Tree, Random Forest

To run Adaboost, type the following:

```bash
python main.py
```

This will run Adaboost on selected features of the dataset and output the score
on every step. A graph will be shown at the end. To change the model, go to line
25 and changed the selected_classifier to one of the other two classifiers.

### R Code

To run Decision Tree, type the following:

```bash
Rscript RPartScript.R fileName
```
fileName - output file from clean.py : complete_data.csv.
This will built a Decision Tree model with all the features.

To run GBM Model, type the following:

```bash
Rscript FullFeatureGBM.R fileName
```
fileName - output file from clean.py : complete_data.csv.
This will built a GBM model with all the features.

To tune GBM Model parameters, type the following:

```bash
Rscript GBMParameterTuning.R fileName
```
fileName - output file from clean.py : complete_data.csv.
This will tune the GBM model paramters with all the features.



