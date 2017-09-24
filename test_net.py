'''
###########################
------------------------------------------------------------------------------------------------------
File name: test_net.py
Created by: Maj Amit Pathania
Roll No: 163054001
Assignment 2
------------------------------------------------------------------------------------------------------
###########################

'''
import numpy as np
import pandas as pd
import pickle

#define sigmoid as activation funtion 
def activationfn(x):
	return 1/(1+np.exp(-x))

#define derivative of activation funtion sigmoid with input as activation function output
def activationfnderivative(x):
	return (x)*(1- x)


#######################################################################################
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
# READING TEST DATA
###########################
test_data=pd.read_csv('kaggle_test_data.csv')
#create copy of test data
o_test_data=test_data
#call pre-defined function to normalise test data
test_data=normalize_data(test_data)
#call pre-defined function to carry out hot encoding on train data
X_test=hot_encoding(test_data)

##########################################################
#Parameters for neural network
################################################################
#get total number of features and test data set rows
total_m,input_layer=X_test.shape
#number of output layers
output_layer=1
#architecture of proposed neural network
neural_net=[input_layer,80,40,20,output_layer]
#list to store output of each layer
l_output=[]
#list to store error of each layer
l_error=[]
#list to store derivative of error for each layer
delta_error=[]
#store length of neural net
L=len(neural_net)
#fill zeros in newly created lists
for i in range(0,L):
	l_output.append(0)
	l_error.append(0)
	delta_error.append(0)


#create list to store wt_matrix
wt_matrix=[]
#Read weights.txt and copy it's contents into wt_matrix
wt_matrix = pickle.load(open('weights.txt', 'rb'))

	

##############################################################
##predicting output for test data
###############################################################

# add the bias to the input layer
ones = np.atleast_2d(np.ones(X_test.shape[0]))
X_test = np.concatenate((ones.T, X_test), axis=1)

#set input as output of first input layer
#l_input=X_test
l_output[0]=X_test
for l in range(1,L):
#input to current layer=output of previous layer
	l_input=l_output[l-1]
	#output of current layer=sigmoid(input*weight)
	l_output[l]=(activationfn(np.dot(l_input,wt_matrix[l-1])))

#output of last layer is predicted value
predict_y=l_output[L-1]

#call pre-defined function to threshold the output of last layer 
predict_y=threshold_output(predict_y)	
#reshape the output
predict_y=np.reshape(predict_y,(len(predict_y),1))

###########################
# WRITING SUBMISSION FILE
###########################
print("Writing to output file......")

o_test_data['salary'] = predict_y.astype(int)
o_test_data[["id","salary"]].to_csv("predictions.csv",index=False)

print("Output file successfully written.")

