'''
###########################
------------------------------------------------------------------------------------------------------
File name: train_standarf.py
Created by: Maj Amit Pathania
Roll No: 163054001
Assignment 2
------------------------------------------------------------------------------------------------------
###########################

'''
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


########################################################################################
#function to normalise input data.The continuous numeric value features were scaled
#########################################################################################

def normalize_data(total_data):
	result = total_data.copy()
	#feature “age” was scaled using factor of 10.. Eg Age=Age/10.
	#This reduced the range of this feature vector between 1-9 (at max).
	result["age"] = ((total_data["age"]) / (10))
	#to reduce wide range of values features ‘fnlwgt’, ‘capital-gain’,'hours-per-week" and ‘capital-loss’, the log was taken 
	result["fnlwgt"]=np.log(1+total_data["fnlwgt"])
	result["capital-gain"]=np.log(1+total_data["capital-gain"])
	result["capital-loss"]=np.log(1+total_data["capital-loss"])
	result["hours-per-week"]=(np.log(1+total_data["hours-per-week"]) )
	
	return(result)


####################################################################################################################################
#function for dummy or hot encoding. Dummy coding is used for converting a categorical input variable into continuous variable. 
#A duplicate variable(new column) which represents one level of a categorical variable. 
#Presence of a level is represent by 1 and absence is represented by 0. For every level present, one dummy variable will be created.
#######################################################################################################################################

def hot_encoding(total_data):
	#define column as category type with given only categories
	total_data["workclass"] = total_data["workclass"].astype('category',categories=[" Federal-gov"," Local-gov"," Never-worked"," Private"," Self-emp-inc"," Self-emp-not-inc"," State-gov"," Without-pay"])
	total_data["marital-status"] =total_data["marital-status"].astype('category',categories=[" Divorced"," Married-AF-spouse"," Married-civ-spouse"," Married-spouse-absent"," Never-married"," Separated"," Widowed"])
	total_data["occupation"] =total_data["occupation"].astype('category',categories=[" Adm-clerical"," Armed-Forces"," Craft-repair"," Exec-managerial"," Farming-fishing"," Handlers-cleaners"," Machine-op-inspct"," Other-service"," Priv-house-serv"," Prof-specialty"," Protective-serv"," Sales"," Tech-support"," Transport-moving"])
	total_data["relationship"] =total_data["relationship"].astype('category',categories=[" Wife"," Own-child"," Husband"," Not-in-family"," Other-relative"," Unmarried"])
	total_data["race"] =total_data["race"].astype('category',categories=[" Amer-Indian-Eskimo"," Asian-Pac-Islander"," Black"," Other"," White"])
	total_data["sex"] =total_data["sex"].astype('category',categories=[" Female"," Male"])
	total_data["native-country"] =total_data["native-country"].astype('category',categories=[" United-States"," Cambodia"," England"," Puerto-Rico"," Canada"," Germany"," Outlying-US(Guam-USVI-etc)"," India"," Japan"," Greece"," South"," China"," Cuba"," Iran"," Honduras"," Philippines"," Italy"," Poland"," Jamaica"," Vietnam"," Mexico"," Portugal"," Ireland"," France"," Dominican-Republic"," Laos"," Ecuador"," Taiwan"," Haiti"," Columbia"," Hungary"," Guatemala"," Nicaragua"," Scotland"," Thailand"," Yugoslavia"," El-Salvador"," Trinadad&Tobago"," Peru"," Hong"," Holand-Netherlands"])
	total_data["education"] =total_data["education"].astype('category',categories=[" Bachelors"," Some-college"," 11th"," HS-grad"," Prof-school"," Assoc-acdm"," Assoc-voc"," 9th"," 7th-8th"," 12th"," Masters"," 1st-4th"," 10th"," Doctorate"," 5th-6th"," Preschool"])

	cats=[" United-States"," Cambodia"," England"," Puerto-Rico"," Canada"," Germany"," Outlying-US(Guam-USVI-etc)"," India"," Japan"," Greece"," South"," China"," Cuba"," Iran"," Honduras"," Philippines"," Italy"," Poland"," Jamaica"," Vietnam"," Mexico"," Portugal"," Ireland"," France"," Dominican-Republic"," Laos"," Ecuador"," Taiwan"," Haiti"," Columbia"," Hungary"," Guatemala"," Nicaragua"," Scotland"," Thailand"," Yugoslavia"," El-Salvador"," Trinadad&Tobago"," Peru"," Hong"," Holand-Netherlands"]

	#bitencode_age=pd.get_dummies(total_data.ix[0:,1],prefix='age_cat')
	bitencode_workclass=pd.get_dummies(total_data.ix[0:,2],prefix='workclass')
	bitencode_education=pd.get_dummies(total_data.ix[0:, 4],prefix='education')
	bitencode_maritalstatus=pd.get_dummies(total_data.ix[0:,6],prefix='maritalstatus')
	bitencode_occupation=pd.get_dummies(total_data.ix[0:, 7],prefix='occupation')
	bitencode_relationship=pd.get_dummies(total_data.ix[0:, 8],prefix='relationship')
	bitencode_race=pd.get_dummies(total_data.ix[0:, 9],prefix='race')
	bitencode_sex=pd.get_dummies(total_data.ix[0:, 10],prefix='', prefix_sep='')
	bitencode_nativecountry=pd.get_dummies(total_data.ix[0:, 14],prefix='', prefix_sep='')
	bitencode_nativecountry = bitencode_nativecountry.reindex(columns=cats, fill_value=0)
	#remove columns containing categorical non-numeric attributes.
	total_data.drop(['id','workclass','education','marital-status','occupation','relationship','race','sex','native-country'],axis=1,inplace=True)
	X_Full = total_data.ix[0:, 0:6]
	#add hot encoded values to the data
	X_Full=pd.concat((X_Full,bitencode_workclass,bitencode_education,bitencode_maritalstatus,bitencode_occupation,bitencode_relationship,bitencode_race,bitencode_sex,bitencode_nativecountry),axis=1)
	#print(X_Full.shape)
	return(X_Full)


#####################################################################################################################################
# Function to threshold the output of last layer 
# It thresholds predicted values.If value is greater than andequal to .5 then output is 1 ,
# otherwise output is zero
#####################################################################################################################################

def threshold_output(output):
	for i in range(0,len(output)):
		if output[i] >= 0.5:
			output[i]=1.0
		else:
			output[i]=0.0
	return(output)

###############################################################################################################################
#
# *****************************************MAIN PROGRAM STARTS HERE***************************************************
#
###########################################################################################################################


###########################
# READING TRAINING DATA
###########################
#read complete data
total_data = pd.read_csv('train.csv')

#Store output Y in variable
Y_Full = total_data.ix[0:, 15]

#call pre-defined function to normalise train data
total_data=normalize_data(total_data)
#call pre-defined function to carry out hot encoding on train data
X_Full=hot_encoding(total_data)


X_train=X_Full
Y_train=Y_Full

##########################################################################
# ***********************MODEL1:Logistic regression********************
###########################################################################
print("Applying Logistic Regression..")
model1 = linear_model.LogisticRegression(fit_intercept=False)
model1.fit(X_train,Y_train)


#########################################################################
# ************************MODEL2:Naive Bayes******************************
##########################################################################
print("Applying Naive Bayes..")
model2 =   GaussianNB()
model2.fit(X_train,Y_train)

###########################################################################
# **************************MODEL3:Decision Tree***************************
##########################################################################

print("Applying Decision Tree......")
model3 = tree.DecisionTreeClassifier()
model3.fit(X_train,Y_train)


###########################
# READING TEST DATA
###########################
test_data=pd.read_csv('kaggle_test_data.csv')
#create copy of test data
o_test_data=test_data
#call pre-defined function to normalise test data
test_data=normalize_data(test_data)
#call pre-defined function to carry out hot encoding on train data
X_test=hot_encoding(test_data)


############################################################
#predicting output for test data using Logistic Regression
############################################################
predict_y=model1.predict(X_test)

print("Writing to output file for Logistic Regression......")

o_test_data['salary'] = predict_y.astype(int)
o_test_data[["id","salary"]].to_csv("predictions_1.csv",index=False)

print("Output file successfully written for Logistic Regression.")

######################################################
#predicting output for test data using Naive Bayes
#####################################################
predict_y=model2.predict(X_test)

print("Writing to output file for Naive Bayes......")

o_test_data['salary'] = predict_y.astype(int)
o_test_data[["id","salary"]].to_csv("predictions_2.csv",index=False)

print("Output file successfully written for Naive Bayes.")


################################################
#predicting output for test data using decision tree
#################################################
predict_y=model3.predict(X_test)

print("Writing to output file  for Decision tree......")

o_test_data['salary'] = predict_y.astype(int)
o_test_data[["id","salary"]].to_csv("predictions_3.csv",index=False)

print("Output file successfully written for Decision tree.")

