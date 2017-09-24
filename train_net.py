'''
###########################
------------------------------------------------------------------------------------------------------
File name: train_net.py
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
#function for dummy or hot encoding. Dummy coding is used for converting a categorical input variable into numeric variable. 
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



##########################################################
#Parameters for neural network
################################################################
#get total number of features and training data set rows
total_m,input_layer=X_train.shape
#number of output layers
output_layer=1
#architecture of proposed neural network
neural_net=[input_layer,80,40,20,output_layer]
#max number of iterations
max_iter=500
#learing rate
learning_rate=.02 
#value of lambda for regularizer
lamda=0.00001 
#condition for convergence
conv_condn=0.00000001 
#input data batch size for training the net
batch_size=500
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

# add the bias to the input layer
ones = np.atleast_2d(np.ones(X_train.shape[0]))
X_train = np.concatenate((ones.T, X_train), axis=1)

#reshaping the Y_train
Y_train=np.reshape(Y_train,(len(Y_train),1))


#intialize weights randomly
wt_matrix=[]
for i in range(1,L-1):
	wt_matrix.append(2*np.random.random((neural_net[i-1]+1,neural_net[i]+1))-1)
wt_matrix.append(2*np.random.random((neural_net[i]+1,neural_net[i+1]))-1)

#define parameters for the loop
converged=False  # in case difference in error between two iterations is less than convergence value,it will set True
cost_f=[] # to store error values for comaprison
diff=1 #initialise variable
#initialise number of iterations to zero
n_iter=0 
while n_iter<max_iter and converged ==False:
	#increment number of iterations
	n_iter=n_iter+1
	index=0
	batch_error = 0 #vriable to store batch_error
	

	for batch_num in range(0,int(total_m/batch_size)):
		start = index*batch_size
		end = (index+1)*batch_size
		#print("Processing batch number : ", batch_num, "Iteration Number: ",n_iter)
		l_input=X_train[start:end]
		l_output[0]=l_input

		#Step1: Feed forward propagation
		for l in range(1,L):
			#print("Forward Propagation:Layer:",l)
			#input to current layer=output of previous layer
			l_input=l_output[l-1]
			#output of current layer=sigmoid(input*weight)
			l_output[l]=(activationfn(np.dot(l_input,wt_matrix[l-1])))
		
		#Step2: Calculate output layer error

		#calculating Error for final output layer (Lth layer)	
		l_error[L-1]= Y_train[start:end] - l_output[L-1]
		
		#if it's first batch then mean_error=mean of current_batch_error
		if (batch_num==0):
			mean_lerror=np.mean(np.abs(l_error[L-1]))
		#else mean_error=previous_mean_error+mean of current_batch_error
		else:
			mean_lerror=mean_lerror+np.mean(np.abs(l_error[L-1]))
		#print(l_error[L-1])
		
		#Step3: Calculate error derivative

		#calculating delta_error for final output layer (Lth layer)
		delta_error[L-1]=l_error[L-1]*activationfnderivative(l_output[L-1])
		#print(delta_error[L-1])

		#Step4: Back propagate the derivative error 

		#Backward propagation from l=L-1 to 2
		for l in range(L-2, 0 ,-1):
			#print("Backward propagation:Layer:  ",l)
			#error of l-1 layer=derivative of error of l layer * weight
			l_error[l] = delta_error[l+1].dot(wt_matrix[l].T)
			#calculate derivate of error for current layer
			delta_error[l] = l_error[l]*activationfnderivative(l_output[l])

		#Step5: Update the weights
		#update weights of the matrix
		for l in range(0,L- 1):
			#update for (l-1)th layer = output of (l-1)th layer* delta_error of (l)th layer
			update=l_output[l].T.dot(delta_error[l+1]) + lamda*wt_matrix[l]/(2*total_m)
			#print(update)
			wt_matrix[l] = wt_matrix[l] + learning_rate*update
		index=index+1
	print("Error after ",n_iter," iterations is ",mean_lerror/index)
	cost_f.append(mean_lerror/index)


	#take difference of error between k and k+1 step
	if(n_iter>2):
		prev=n_iter-2
		curr=n_iter-1
		diff=(cost_f[prev]-cost_f[curr])
	#check for convergence
	if(abs(diff)<conv_condn):
		converged=True
		print("converged")


#dump weight vector in file
pickle.dump(wt_matrix, open('weights.txt', 'wb'))
print("Training completed")