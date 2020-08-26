import pandas as pd
import numpy as np
from scipy.spatial import distance
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
pandas2ri.activate()
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.conversion import localconverter
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
pandas2ri.activate()
from matplotlib import pyplot
from collections import Counter
from sklearn.model_selection import StratifiedKFold, KFold
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler, PowerTransformer, StandardScaler
from scipy.stats import normaltest
from sklearn.model_selection import ParameterSampler
from numpy.random import randn
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from numpy import *
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats.stats import pearsonr, spearmanr
from scipy import stats
import tensorflow as tf
import multiprocessing as mp
import time
import os
import collections
import matplotlib.pyplot as plt
import itertools as it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import re
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
                                            #Setting Global Variables
########################################################################################################################

#specify path of dataset
input_path = "/home/yhh05/smogN/Manual_Daily_Albedo_NDVI_LST_Cleaned.csv"

#specify saved file directory 
output_path = "/home/yhh05/smogN/output/"

#specify columns to drop
columnsToDrop = ['Date','Year','Month','Day',
                 'Month_1', 'Month_2', 'Month_3', 'Month_4',
                 'Vegetation_1', 'Vegetation_2','Vegetation_3',
                 'Climate', 'Vegetation', 'Latitude', 'Longitude',
                 'G','G-1','G-2','G-3','G-4','G-5',
                 'Climate_1', 'Climate_2', 'Climate_3',
                 'Latitude_1','Latitude_2', 'Latitude_3', 'Latitude_4', 'Latitude_5',
                 'Latitude_6','Longitude_1', 'Longitude_2', 'Longitude_3', 'Longitude_4',
                 'Longitude_5', 'Longitude_6',
                 'H', 'H_bowen_corr', 'H_bowen_corr-1', 'H_bowen_corr-2', 'H_bowen_corr-3', 'H_bowen_corr-4',
                 'H_bowen_corr-5', 'C_BOWENS',
                 'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
                 'LE', 'LE_bowen_corr',
                 'Elevation(m)_1','Elevation(m)_2', 'Elevation(m)_3', 'Elevation(m)_4',
                 'Elevation(m)_5', 'Elevation(m)_6',
                 'ETo', 'EToF', 'ETr', 'ETrF',
                'Site Id_1', 'Site Id_2', 'Site Id_3', 'Site Id_4', 'Site Id_5',
                 'Site Id_6']           

#specify the output column
output_column = "LE_bowen_corr_mm"

#specify name of target variable and name of predicted target variable
y_test_name = 'LE_bowen_corr_mm'
y_test_pred_name = 'LE_bowen_corr_mm_pred'

#rename variables with spacing and under score for better proper namings
columns_rename={"Site Id_1": "Site_1", "Site Id_2": "Site_2",
         "Site Id_3": "Site_3", "Site Id_4": "Site_4",
         "Site Id_5": "Site_5", "Site Id_6": "Site_6"}

#specify one-hot encoded vector names 
one_hot_encoded = []
                  
#specify desired split size
test_size = 0.2

#specify if scaling
scaling = True

#specify if automatic or manual scaling
automatic = True
  
#if not automatic specify desired column names
all_columns = ['WS', 'RH', 'TA', 'LE', 'ET_bowen_corr']
#specify the scaling type for each column
scaling = ['MinMax', 'Standard', 'Robust', 'MinMax + PowerTransform', 'Standard + PowerTransform']

#specify the option of utility based
utility_based = True

#specify if batch training 
train_batches = False

#specify number of parameters in random search
n_params = 100

#specify batch size of hyper-parameters
batch_size = 4

#specify number of batch you'd like to train model over
batch_num = 2

#specify if random search
random_search = False

#specify if grid search
grid_search = False 

#spcify repetitions and folds for repeated stratified cross validation 
repetitions = 1
folds = 5

#specify if you wish to apply over sampling by smogn
smogn = True

#smogn relate hyper-params
target_variable = "LE_bowen_corr_mm"
rel_method='range'
extr_type='high'
coef=1.5
rell = np.array([
    [1, 0 , 0],
    [9, 0 , 0],
    [15 ,1, 0]
])
#rell = None
relevance_pts=rell
rel="auto"
thr_rel=0.5
Cperc=np.array([1,150])
k=5
repl=False
dist="Euclidean"
p=2
pert=0.1

########################################################################################################################
                                            #Helper Methods
########################################################################################################################
def get_rare(y, method, extr_type, thresh, coef, control_pts):

    # we will be getting the relevance function on all the data not just the training data because
    # when we want to apply Lime on the 'rare' testing instances, the relevance function must map all possible demand
    # values to a certain relevance. If it happens that some demand values are present only in the testing
    # and not in the training data, we cannot detect rare values correctly. The way we compute
    # rare values depends on the relevance

    # param y: the target variable vector
    # param method: 'extremes' or 'range'. Default is 'extremes'
    # param extr_type: 'both', 'high', or 'low'
    # param thresh: threshold. Default is 0.8
    # param coef: parameter needed for method "extremes" to specify how far the wiskers extend to the most extreme data point in the boxplot. The default is 1.5.
    # param control_pts: if method == 'range', then this is the relevance matrix provided by the user. Default is None

    # return the indices of the rare values in the data

    yrel = get_relevance_2(y, df=None, target_variable=None, method=method, extr_type=extr_type, control_pts=control_pts)

    # get the the phi.control returned parameters that are used as input for computing the relevance function phi
    # (function provided by R UBL's package: https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi)
    # (function provided by R UBL's package
    # https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi.control)
    # we need those returned parameters for computing rare values

    # print('relevance method - phi function : {}'.format(method))

    if control_pts is None:
        # without relevance matrix
        # print('control.pts - phi function: {}'.format(control_pts))
        # print('without relevance matrix')
        params = runit.get_relevance_params_extremes(y, rel_method=method, extr_type=extr_type, coef=coef)
    else:
        # with relevance matrix (provided by the user)
        # print('control.pts - phi function: {}'.format(control_pts))
        # print('with relevance matrix')
        params = runit.get_relevance_params_range(y, rel_method=method, extr_type=extr_type, coef=coef,
                                                  relevance_pts=control_pts)

    # phi params
    phi_params = params[0]
    loss_params = params[1]

    phi_params = dict(zip(phi_params.names, list(phi_params)))
    loss_params = dict(zip(loss_params.names, list(loss_params)))

    # print('\nCONTROL PTS')
    # print(phi_params['control.pts'])
    print("for the whole dataset")
    rare_indices = get_rare_indices(y=y, y_rel=yrel, thresh=thresh, controlpts=phi_params['control.pts'])
    # print('rare indices are: {}'.format(rare_indices))

    return rare_indices, phi_params, loss_params, yrel


def get_relevance_2(y, df, target_variable, method, extr_type, control_pts):

    # gets the relevance values of the target variable vector
    # param y: the target variable vector
    # param df: if y in None, this must be passed. It is the data frame of interest
    # param target_variable: if y is None, this must be passed. It is the name of the target variable
    # param method: 'extremes' or 'range'
    # param extr_type: 'both', 'high', or 'low'
    # param control_pts: if method == 'range', will be a relevance matrix provided by the user
    # return: the relevance values of the associated target variable

    # get the target variable vector y
    if y is None:
        if df is None or target_variable is None:
            raise ValueError('if y is None, neither df nor target_variable must be None')
        y = df[target_variable]

    # check that the passed parameters are in order
    if method != 'range' and method != 'extremes':
        raise ValueError('method must be "range" or "extremes", there is no method called "%s"' % method)
    elif method == 'range' and control_pts is None:
        raise ValueError('If method == "range", then control_pts must not be None')
    elif method == 'extremes' and extr_type not in ['high', 'low', 'both']:
        raise ValueError('extr_type must wither be "high", "low", or "both"')
    else:
        if control_pts is None:
            # print('getting yrel - Control pts is {}, method is {}'.format(control_pts, method))
            y_rel = runit.get_yrel(y=np.array(y), meth=method, extr_type=extr_type)
        else:
            # print('getting yrel - Control pts is not None, method is {}'.format(method))
            y_rel = runit.get_yrel(y=np.array(y), meth=method, extr_type=extr_type, control_pts=control_pts)

    return y_rel


def get_rare_indices(y, y_rel, thresh, controlpts):
    # get the indices of the rare values in the data
    # param y: the target variable vector
    # param y_rel: the target variable (y) relevance vector
    # param thresh: the threshold of interest
    # param controlpts: the phi.control (function provided by R UBL's package: https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi.control)
    # returned parameters that are used as input for computing the relevance function phi (function provided by R UBL's package: https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/phi)
    # return: the indices of the rare values in 'y'
    

    # references
    # https://github.com/paobranco/SMOGN-LIDTA17/blob/8964a2327de19f6ca9e6f7055479ca863cd6b8a0/R_Code/ExpsDIBS.R#L41

    # transform controlpts returned by R into a python list
    controlpts = list(np.array(controlpts))
    # print(controlpts)

    # boolean variable indicating whether both low and high rare exist
    both = [controlpts[i] for i in [1, 7]] == [1, 1]

    # initialize rare cases to empty list (in case there are no rare cases at all)
    rare_cases = []

    if both:
        # bothr = True
        print('\nWe have both low and high extremes')
        rare_low = [i for i, e in enumerate(y_rel) if e > thresh and y[i] < controlpts[3]]
        rare_high = [i for i, e in enumerate(y_rel) if e > thresh and y[i] > controlpts[3]]

        # merge two lists (of low rare + high rare) together
        rare_cases = rare_low + rare_high

    else:
        print('\nWe dont have both', end=' ')
        if controlpts[1] == 1:
            print('We have only low rare')
            # lowr = True
            rare_cases = [i for i, e in enumerate(y_rel) if e > thresh and y[i] < controlpts[3]]
        else:
            print('We have only high rare')
            # highr = True
            rare_cases = [i for i, e in enumerate(y_rel) if e > thresh and y[i] > controlpts[3]]

    total = len(rare_cases)

    print('Total Number of rare cases: %d out of %d' % (total, len(y)))
    print('Percentage of Rare Cases: %.2f%%\n' % (total/len(y) * 100))

    return rare_cases

def rarify_data(df, df_train, df_test, target_variable, method, extr_type, thresh, coef, control_pts):

    # get df_train and df_test
    # param df_train: the training data frame
    # param df_test: the testing data frame
    # param target_variable: name of the target variable column
    # return: df_train and df_test with equal class distribution between classes: rare and not rare

    # print("checking null values in dataset when applying rarify")
    # print(df.isnull().values.any())

    # get y, reset the index to avoid falsy retrievals by index later on
    y = df[target_variable].reset_index(drop=True)
    # get the indices of the rare values in the combined data frame
    # note that the relevance returned is the relevance of the whole data frame not just the training
    rare_values, phi_params, loss_params, yrel = get_rare(y, method, extr_type,thresh, coef, control_pts)

    # dictionary mapping each value to its relevance
    demandrel = {}
    relvals = np.array(yrel)

    for i, e in enumerate(y):
        if e not in demandrel:
            rel = relvals[i]
            demandrel[e] = rel

    # now we have the indices of the rare values, get their percentage

    # percentage of rare values in the whole dataset
    prare = len(rare_values)/len(df)
    print('percentage of rare values in dataset before smogn: ' + str(prare*100) , file=open(output_path +"rare_perc_results.txt", "a"))
    # number of rare values in the whole dataset
    numrare = len(rare_values)
    print('number of rare values in dataset before smogn: {}/{}'.format(numrare, len(df)), file=open(output_path +"rare_perc_results.txt", "a"))

    # number of rare values that must be in each of the train and test
    numraretrain = int(round(prare * len(df_train)))
    numraretest = int(round(prare * len(df_test)))

    print('number of rare that are  in train: {}/{}'.format(numraretrain, len(df_train)))
    print('==> {}%%'.format((numraretrain/len(df_train))*100))
    print('number of rare that are in test: {}/{}'.format(numraretest, len(df_test)))
    print('==> {}%%'.format((numraretest / len(df_test))*100))

    rare_values = sorted(rare_values)

    # rare indices partitioned for each of the train and test
    rtrain = rare_values[:numraretrain]
    rtest = rare_values[numraretrain:]

    # get the relevance of each of the  dftrain and dftest
    yreltrain = [demandrel[d] for d in df_train[target_variable]]
    yreltest = [demandrel[d] for d in df_test[target_variable]]

    # if len(rtrain) != numraretrain:
    #     raise ValueError('Incompatibility between the number of rare values that must be included in the '
    #                      'training data for equal class distribution and the obtained number of rare')

    # if len(rtest) != numraretest:
    #     raise ValueError('Incompatibility between the number of rare values that must be included in the '
    #                      'testing data for equal class distribution and the obtained number of rare')

    return df_train, df_test, rtrain, rtest, yreltrain, yreltest, phi_params['control.pts'], loss_params, demandrel


def get_phi_loss_params(y, rel_method, extr_type='high', coef=1.5, relevance_pts=None):

    # get the parameters of the relevance function
    # param df: dataframe being used
    # param target_variable: name of the target variable
    # param rel_method: either 'extremes' or 'range'
    # param extr_type: either 'high', 'low', or 'both' (defualt)
    # param coef: default: 1.5
    # param relevance_pts: the relevance matrix in case rel_method = 'range'
    # return: phi parameters and loss parameters


    if relevance_pts is None:
        print('Will not use relevance matrix')
        params = runit.get_relevance_params_extremes(y, rel_method=rel_method, extr_type=extr_type, coef=coef)
    else:
        print('Using supplied relevance matrix')
        params = runit.get_relevance_params_range(y, rel_method=rel_method, extr_type=extr_type, coef=coef,
                                                  relevance_pts=relevance_pts)

    # phi params and loss params
    phi_params = params[0]
    loss_params = params[1]
    relevance_values = params[2]

    phi_params = dict(zip(phi_params.names, list(phi_params)))
    loss_params = dict(zip(loss_params.names, list(loss_params)))

    return phi_params, loss_params, relevance_values


def get_relevance_oversampling(smogned, target_variable, targetrel):

    # gets the relevance values of an oversampled data frame
    # param smogned: the oversampled data frame
    # param target_variable: name of the target variable column
    # param targetrel: dictionary mapping each target variable value to a relevance value
    # return: the relevance of the oversampled data frame

    yrelafter = []
    distances = []
    for val in smogned[target_variable]:
        if val in targetrel:
            yrelafter.append(targetrel[val])
        else:
            nearest = min(sorted(list(targetrel.keys())), key=lambda x: abs(x - val))
            distances.append(abs(nearest - val))
            yrelafter.append(targetrel[nearest])

    return yrelafter, distances
  
def get_formula(target_variable):

    # gets the formula for passing it to R functions. Example: target_variable ~ col1 + col2 ...
    # param target_variable: the name of the target variable
    # return: R's formula as follows: target_variable ~ other[0] + other[1] + other[2] + other[3] + ...

    formula = runit.create_formula(target_variable)
    return formula
    
def write_to_txt(filename, content):
  text_file = open(output_path + filename, "w")
  text_file.write(content)
  text_file.close()

def get_relevance():
    ctrl = phi_params['control.pts']
    if rel_method[0] == 'extremes' and relevance_pts[0] is None:
        rell = np.array([
            [ctrl[0], ctrl[1], ctrl[2]],
            [ctrl[3], ctrl[4], ctrl[5]],
            [ctrl[6], ctrl[7], ctrl[8]]
        ])
    else:
        rell = relevance_pts[0]

    return rell

## Generate lags for all input features, re-generate even if some exist so that order will not be shuffled after nan dropping
def generate_lags_for(df, column, lags_count):
        for i in range(lags_count):
            lag_name = column + "-" + str(i + 1)
            df[lag_name] = df[column].shift(i + 1)
        return df

def generate_lags(df, lagsForColumns):
    '''This function generates the lags for the list of columns'''
    for k in range(len(lagsForColumns)):
        col = lagsForColumns[k]
        if col in df.columns:
            df = generate_lags_for(df, col, 5)
    return df


def split_train_test_valid(df, site, TRAIN_RATIO, TEST_RATIO):
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_train = pd.DataFrame()
    Y_test = pd.DataFrame()
    
    #print("Number of sites:", len(unique_sites))
    #print(df)
    #print(site)
    df_site = df[df["Site Id"] == site]
    #print(df_site)
    X = df_site
    train_index = int(X.shape[0] * TRAIN_RATIO)
    test_index = int(X.shape[0] * (TRAIN_RATIO + TEST_RATIO))

    X_train = X_train.append(X[:train_index], ignore_index = True)
    X_test = X_test.append(X[train_index:], ignore_index = True)
    Y_train = Y_train.append(X[:train_index], ignore_index = True)
    Y_test = Y_test.append(X[train_index:], ignore_index = True)
   
    Y_train = Y_train[[output_column]]
    Y_test = Y_test[[output_column]]
   
    X_train = X_train.drop([output_column], axis = 1)
    X_test = X_test.drop([output_column], axis = 1)
   
    return X_train, X_test, Y_train, Y_test


########################################################################################################################
                                            #Establish a connection to R library
########################################################################################################################
rpy2.robjects.numpy2ri.activate()
runit = robjects.r
runit['source']('/home/yhh05/smogN/smogn.R')

########################################################################################################################
                                            #Read and Preprocess Dataset
########################################################################################################################

#read csv
df = pd.read_csv(input_path, delimiter=',')


#drop NaN values
df.dropna(inplace=True)

# filter sites if ameri or euro
#df = df[df["Site Id"].str.startswith('US-')]
#df = df[~df["Site Id"].str.startswith('US-')]

#set output variable between 1 and 15 only
df = df[df[output_column].between(1, 15)]

#drop desired columns, rename, and drop the nans
df = df.drop(columnsToDrop, axis = 1)
df.rename(columns_rename, inplace=True)
df.dropna(inplace=True)

#generate lags for columns
lagsForColumns = ["SW_IN", "WS", "RH", "TA", "EEflux LST", "EEflux Albedo", "EEflux NDVI"]
df = generate_lags(df, lagsForColumns)
# print(df.columns)

#drop nan for the first 5 rows of the generated lags only 5 rows will be removed in here
df.isnull().mean() * 10
df.dropna(inplace=True)
print(df.shape)

# print("checking null values in the whole dataset")
# print(df.isnull().values.any())
unique_sites = df["Site Id"].unique()
# print("Number of sites:", len(unique_sites))

for site in unique_sites:

    print("Demonstrating info for Ameriflux site: " + site)
    #split into train and test according to special split
    X_train, X_test, y_train, y_test = split_train_test_valid(df, site, 0.8, 0.2)
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    #dropping site id after filtering the sites
    columnToDrop = "Site Id"
    X_train.drop([columnToDrop], axis = 1, inplace=True)
    X_test.drop([columnToDrop], axis = 1, inplace=True)

    #defining the test dataset
    df_test = X_test
    df_test[y_test_name] = y_test
    
    #defining train dataset
    df_train = X_train
    df_train[output_column] = y_train
    size_original = df_train.shape

    #resetting indexes 
    df_train.reset_index(drop=True)
    df_test.reset_index(drop=True)

    # #checking if null values exist here
    # print("checking null values in train")
    # print(df_train.isnull().values.any())
    # print("checking null values in test")
    # print(df_test.isnull().values.any())

    #retrieve indexes of rare values to utilize in stratified folds
    df_train_rare, df_test_rare, rtrain, rtest, yreltrain, yreltest, phi_params, loss_params, targetrel = rarify_data(df, df_train, df_test, output_column, rel_method,extr_type, thr_rel,coef, relevance_pts)