import pandas as pd
pd.set_option('use_inf_as_na', True)
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
from sklearn.metrics.cluster import normalized_mutual_info_score
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
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools as it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import re
import warnings
import category_encoders as ce
warnings.filterwarnings("ignore")

########################################################################################################################
                                            #Setting Global Variables
########################################################################################################################

#specify path of dataset
input_path = "/home/yhh05/smogN/joint_library_cleaned.csv"

#specify saved file directory 
output_path = "/home/yhh05/smogN/output/"

#specify columns to drop
##Manual

##columnsToDrop = ['Date','Year','Month','Day',
#                  'Climate', 'Vegetation', 'Latitude', 'Longitude',
#                  'G','G-1','G-2','G-3','G-4','G-5',
#                  'Climate_1', 'Climate_2', 'Climate_3',
#                  'Latitude_1','Latitude_2', 'Latitude_3', 'Latitude_4', 'Latitude_5',
#                  'Latitude_6','Longitude_1', 'Longitude_2', 'Longitude_3', 'Longitude_4',
#                  'Longitude_5', 'Longitude_6',
#                  'H', 'H_bowen_corr', 'H_bowen_corr-1', 'H_bowen_corr-2', 'H_bowen_corr-3', 'H_bowen_corr-4',
#                  'H_bowen_corr-5', 'C_BOWENS',
#                  'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
#                  'LE', 'LE_bowen_corr',
#                  'Elevation(m)_1','Elevation(m)_2', 'Elevation(m)_3', 'Elevation(m)_4',
#                  'Elevation(m)_5', 'Elevation(m)_6',
#                  'ETo', 'EToF', 'ETr', 'ETrF']


#Manual Joint 
#columnsToDrop = ['Date','Year','Month','Day',
#                  'Climate', 'Vegetation', 'Latitude', 'Longitude',
#                  'G','G-1','G-2','G-3','G-4','G-5',
#                  'Climate_1', 'Climate_2', 'Climate_3',
#                  'Latitude_1','Latitude_2', 'Latitude_3', 'Latitude_4', 'Latitude_5',
#                  'Latitude_6','Longitude_1', 'Longitude_2', 'Longitude_3', 'Longitude_4',
#                  'Longitude_5', 'Longitude_6',
#                  'H', 'H_bowen_corr', 'H_bowen_corr-1', 'H_bowen_corr-2', 'H_bowen_corr-3', 'H_bowen_corr-4',
#                  'H_bowen_corr-5', 'C_BOWENS',
#                  'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
#                  'LE', 'LE_bowen_corr',
#                  'Elevation(m)_1','Elevation(m)_2', 'Elevation(m)_3', 'Elevation(m)_4',
#                  'Elevation(m)_5', 'Elevation(m)_6', 'EToF', 'ETr', 'ETrF']
#                  
                  
#                  
#Library
#columnsToDrop = ['Date','Year','Month','Day','Latitude','Longitude',
#                 'Climate','Vegetation', 'G','G-1','G-2','G-3','G-4','G-5',
#                 'H','H_bowen_corr','H_bowen_corr-1','H_bowen_corr-2','H_bowen_corr-3',
#                 'H_bowen_corr-4','H_bowen_corr-5', 'H_ebr_corr','H_ebr_corr-1','H_ebr_corr-2',
#                 'H_ebr_corr-3','H_ebr_corr-4','H_ebr_corr-5','LE_ebr_corr','LE_ebr_corr(mm)',
#                 'ET_bowen','ET_bowen_corr','ET_bowen_corr(mm)','ET_ebr','ET_ebr_corr',
#                 'ET_ebr_corr(mm)' ,'NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4',
#                 'NETRAD-5','LE','LE_bowen_corr', 'ETo','EToF_bowen','EToF_ebr',
#                 'ETr','ETrF_bowen','ETrF_ebr', 'Climate_1',
#                 'Climate_2','Climate_3', 'Latitude_1',
#                 'Latitude_2','Latitude_3','Latitude_4','Latitude_5','Latitude_6', 'Longitude_1',
#                 'Longitude_2','Longitude_3','Longitude_4','Longitude_5','Longitude_6',
#                 'Elevation(m)_1','Elevation(m)_2','Elevation(m)_3','Elevation(m)_4',
#                 'Elevation(m)_5','Elevation(m)_6', 'NETRAD', 'Site_1', 'Site_2', 'Site_3', 'Site_4', 'Site_5', 'Site_6', 'Month_1',
#                 'Month_2', 'Month_3', 'Month_4', 'Vegetation_1', 'Vegetation_2',
#                 'Vegetation_3', 'SW_IN']  
                 
#Joint Lib
#columnsToDrop = ['Day', 'Date',
#                 'Climate', 'Latitude', 'Longitude',
#                 'G','G-1','G-2','G-3','G-4','G-5',
#                 'H', 'H_bowen_corr', 'H_bowen_corr-1', 'H_bowen_corr-2', 'H_bowen_corr-3', 'H_bowen_corr-4',
#                 'H_bowen_corr-5',
#                 'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
#                 'LE', 'LE_bowen_corr', 'Cloud',
#                 'Tier', 'ETr',
#                 'SW_IN',  'H_ebr_corr', 'H_ebr_corr-1', 'H_ebr_corr-2', 'H_ebr_corr-3',
#                 'H_ebr_corr-4', 'H_ebr_corr-5', 'LE_ebr_corr',
#                 'ET_ebr_corr', 'ET_ebr_corr(mm)', 'ET_bowen_corr',
#                 'ET_bowen_corr(mm)', 'ETo', 'EToF_bowen', 'EToF_ebr', 'ETrF_bowen',
#                 'ETrF_ebr', 'EEflux ETo', 'EEflux ETr',
#                 'EEflux EToF', 'EEflux ETrF', 'SW_IN', 'LE_ebr_corr(mm)', 'LE_bowen_corr_mm']  

#columnsToDrop = ['Day', 'Date', 
#                 'Climate', 'Latitude', 'Longitude',
#                 'G','G-1','G-2','G-3','G-4','G-5',
#                 'H', 'H_bowen_corr', 'H_bowen_corr-1', 'H_bowen_corr-2', 'H_bowen_corr-3', 'H_bowen_corr-4',
#                 'H_bowen_corr-5',
#                 'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
#                 'LE', 'LE_bowen_corr', 'Cloud',
#                 'Tier', 'ETr',
#                 'SW_IN', 'EEflux ETo', 'EEflux ETr',
#                 'EEflux EToF', 'EEflux ETrF', 'SW_IN', 'ETo',
#                 'EToF', 'ETrF', 'Image Id']   

#columnsToDrop = ['Year','Day', 'Date',
#                 'Climate', 'Latitude', 'Longitude',
#                 'G','G-1','G-2','G-3','G-4','G-5',
#                 'H', 'H_bowen_corr', 'H_bowen_corr-1', 'H_bowen_corr-2', 'H_bowen_corr-3', 'H_bowen_corr-4',
#                 'H_bowen_corr-5',
#                 'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
#                 'LE', 'LE_bowen_corr', 'Cloud',
#                 'Tier', 'ETr',
#                 'SW_IN', 'ETo','EToF', 'ETrF', 'Image Id',
#                  'Eeflux_ET', 'Eeflux_ETo', 'Eeflux_ETr',
#                 'Eeflux_EToF', 'Eeflux_ETrF']    

columnsToDrop = ['Date', 'LE_bowen_corr_mm']
#specify the output column
output_column = "Eeflux_ET"

#specify name of target variable and name of predicted target variable
y_test_name = 'Eeflux_ET'
y_test_pred_name = 'Eeflux_ET_pred'

#rename variables with spacing and under score for better proper namings
#columns_rename={"Site Id_1": "Site_1", "Site Id_2": "Site_2",
#         "Site Id_3": "Site_3", "Site Id_4": "Site_4",
#         "Site Id_5": "Site_5", "Site Id_6": "Site_6"}

#specify one-hot encoded vector names 
one_hot_encoded = ['Site_1', 'Site_2', 'Site_3', 'Site_4', 'Site_5', 'Month_1',
                 'Month_2', 'Month_3', 'Month_4', 'Vegetation_1', 'Vegetation_2',
                'Vegetation_3']
                  
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
target_variable = "Eeflux_ET"
rel_method='range'
extr_type='high'
coef=1.5

rell = np.array([
    [1, 0 , 0],
    [4, 0 , 0],
    [15, 1 , 0],
])

#rell = None
relevance_pts=rell
rel="auto"
thr_rel=0.1
Cperc=np.array([1,1.2])
k=5
repl=False
dist="Euclidean"
p=2
pert=0.1

########################################################################################################################
                                            #Helper Methods
########################################################################################################################

########################################################################################################################
                                            #Helper Methods
########################################################################################################################

# checks if the input is gaussian by shapiro wilk test
def check_distribution_shapiro(col):
    stat, p = shapiro(col)
    alpha = 0.05
    if p > alpha:
        gaussian = True
    else:
        gaussian = False

    return gaussian

# checks if the input is gaussian by dagostino test
def check_distribution_dagostino(col):
    stat, p = normaltest(col)
    alpha = 0.05
    if p > alpha:
        gaussian = True
    else:
        gaussian = False

    return gaussian


# splits data into train and test
def train_test_splitting(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)
    return X_train, X_test, y_train, y_test


# does automatic standardization according to column values
def standardization_needed(col, X):
    col = col.values.reshape(-1, 1)
    X = X.values.reshape(-1, 1)
    # checks if column has zero values
    if 0 not in col:
        col_trans = Standard(col, X)
        col_trans_pow = apply_power_trans(col_trans)
    else:
        # if there are zero values, applies MinMaxScaler
        # check range of values
        col_trans = Min_Max(col, X)
        col_trans_pow = apply_power_trans(col_trans)
    return np.ravel(col_trans_pow)

# does manual standardization according to user input
def standardization_needed_manual(col, X, scaling):
    col = col.values.reshape(-1, 1)
    X = X.values.reshape(-1, 1)
    if scaling == 'None':
        col_trans_final = col
    if scaling == 'MinMax':
        col_trans_final = Min_Max(col, X)
    if scaling == 'Standard':
        col_trans_final = Standard(col, X)
    if scaling == 'Robust':
        col_trans_final = Robust(col, X)
    if scaling == 'MinMax + PowerTransform':
        col_trans = Min_Max(col, X)
        col_trans_final = apply_power_trans(col_trans)
    if scaling == 'Standard + PowerTransform':
        col_trans = Standard(col, X)
        col_trans_final = apply_power_trans(col_trans)
    return np.ravel(col_trans_final)


def Min_Max(col, X):
    if any(n < 0 for n in col):
        scaler = MinMaxScaler((-1, 1))
    else:
        scaler = MinMaxScaler((0, 1))
    scaler.fit(X)
    col_trans = scaler.transform(col)
    return col_trans

def Min_Max_auto(X_train, X_test, cols):
  for col in X_train.columns:
    if any(n < 0 for n in cols):
      scaler = MinMaxScaler((-1, 1))
    else:
      scaler = MinMaxScaler((0, 1))
    X_train[col] = X_train[col].values.reshape(-1, 1)
    X_test[col] = X_test[col].values.reshape(-1, 1)
    scaler.fit( X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
  return X_test


def Standard(col, X):
    if 0 not in col:
        # if no zero values, apply StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)
        col_trans = scaler.transform(col)
        return col_trans
    else:
        print('column has 0 values, cannot apply standard scaling')
        return col


def Robust(col, X):
    scaler = RobustScaler()
    scaler.fit(X)
    col_trans = scaler.transform(col)
    return col_trans

# does power transform on column automatically
def apply_power_trans(col_trans):
    if any(n <= 0 for n in col_trans):
        # if there are negative or zero values, applies yeo-johnson transform
        pt = PowerTransformer('yeo-johnson')
        col_trans_pow = pt.fit_transform(col_trans)
        # checks if column values are strictly positive
    elif all(n > 0 for n in col_trans):
        # if values are strictly positive, applies box-cox transform
        pt = PowerTransformer('box-cox')
        col_trans_pow = pt.fit_transform(col_trans)
    return col_trans_pow

# does scaling on dataset automatically
def apply_scaling(df, columns, X_train):
    df_scaled = pd.DataFrame(columns=df.columns)
    for i in df.columns: 
        #checking if data is Gaussian
        if i in columns:
            if not check_distribution_shapiro(df[i]) and not check_distribution_dagostino(df[i]):
                print(str(i) + ' does not have a Gaussian distribution and will be scaled')
                #scaling data
                df_scaled[i] = standardization_needed(df[i], X_train[i])
            else: 
                df_scaled[i] = df[i]
                print(str(i) + ' has a gaussian distribution')
        else:
            df_scaled[i] = df[i]
    return df_scaled

# does scaling on dataset according to user input
def apply_scaling_manual(df, columns, X_train, scaling):
    iterator = 0
    if len(scaling) != len(columns):
        print("Please specify scaling for all columns listed")
        return
    else:
        df_scaled = pd.DataFrame(columns=df.columns)
        for i in df.columns:
            if i in columns:
             # checking if data is Gaussian
                if not check_distribution_shapiro(df[i]) and not check_distribution_dagostino(df[i]):
                    print(str(i) + ' does not have a Gaussian distribution and will be scaled')
                    # scaling data
                    df_scaled[i] = standardization_needed_manual(df[i], X_train[i], scaling[iterator])
                    iterator = iterator + 1
                else:
                    df_scaled[i] = df[i]
                    print(str(i) + ' has a gaussian distribution')
            else:
                df_scaled[i] = df[i]
        return df_scaled


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

    print('relevance method - phi function : {}'.format(method))

    if control_pts is None:
        # without relevance matrix
        print('control.pts - phi function: {}'.format(control_pts))
        print('without relevance matrix')
        params = runit.get_relevance_params_extremes(y, rel_method=method, extr_type=extr_type, coef=coef)
    else:
        # with relevance matrix (provided by the user)
        print('control.pts - phi function: {}'.format(control_pts))
        print('with relevance matrix')
        params = runit.get_relevance_params_range(y, rel_method=method, extr_type=extr_type, coef=coef,
                                                  relevance_pts=control_pts)

    # phi params
    phi_params = params[0]
    loss_params = params[1]

    phi_params = dict(zip(phi_params.names, list(phi_params)))
    loss_params = dict(zip(loss_params.names, list(loss_params)))

    print('\nCONTROL PTS')
    print(phi_params['control.pts'])
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
            print('getting yrel - Control pts is {}, method is {}'.format(control_pts, method))
            y_rel = runit.get_yrel(y=np.array(y), meth=method, extr_type=extr_type)
        else:
            print('getting yrel - Control pts is not None, method is {}'.format(method))
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


def round_oversampled_one_hot_encoded(df):

    # round one hot encoded vectors of an oversampled dataset. We have fed the SMOGN/SMOTER/GN/RandUnder
    # a data frame having one hot encoded values (0s and 1s). However, given that we are using Euclidean/Manhattan
    # distances for oversampling, some noise is added to these making them 1.0003, 0.99, etc.
    # Having this said, this function will round these values back again so they are
    # perfect 0s or 1s. We could have used HEOM distance, but it expects "nominal" features
    # as opposed to one hot encodings.
    # param df: the over-sampled data frame
    # return: the over-sampled data frame with one hot encodings rounded

    for col in one_hot_encoded:
        df.loc[df[col] < 0.5, col] = 0
        df.loc[df[col] >=0.5, col] = 1
    return df


def count_abnormal(df):

    # Due to Oversampling, SMOGN is adding noise to the one hot encoded vectors. This function counts how many of these
    # are being done
    # param df: the oversampled data frame
    # return: statistics about the above

    count = 0
    for col in one_hot_encoded:
        for i, row in df.iterrows():
            if row[col] not in [0, 1]:
                count += 1
            else:
                continue

    print('number of noisy one hot encoded: {} out of {}'.format(count, len(df)))
    print('percentage of noisy one hot encoded: %.3f' % (count / len(df) * 100))

#calculates all error metrics needed
def calculate_errors(actual, predicted, nb_columns):
    n = len(actual)
    r2score = r2_score(actual, predicted)
    adjusted_r2 = 1 - ((1 - r2score) * (n - 1)) / (n - nb_columns - 1)
    mase = mean_absolute_error(actual, predicted)
    rms = sqrt(mean_squared_error(actual, predicted))
    mse = mean_squared_error(actual, predicted)
    re = (mse / np.mean(predicted)) * 100
    pearson, pval = stats.pearsonr(np.array(actual).ravel(), np.array(predicted).ravel())
    mae = np.mean(np.abs((actual - predicted) / actual)) * 100
    spearman_corr, _ = spearmanr(actual, predicted)
    distance_corr = distance.correlation(actual, predicted)
    mape_score = np.asarray(np.abs(( np.array(actual) - np.array(predicted)) / np.array(actual)), dtype=np.float64).mean() * 100
    return r2score, mase, rms, mse, re, pearson, pval, mae, adjusted_r2, spearman_corr, distance_corr, mape_score

#get indices of folds in Stratified KFold CV
def get_fold_indices(X,y,n_splits,rare_values):
    rare_vec = [1 if i in rare_values else 0 for i in range(len(y))]
    y = np.array(rare_vec)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=123)
    folds = list(splitter.split(X, y))
    return folds
    
#get indices of folds in Stratified KFold CV
def get_rare_idex(X,y,rare_values):
    rare_vec = [1 if i in rare_values else 0 for i in range(len(y))]
    y = np.array(rare_vec)
    return y
  
#get grid of all hyper-parameters
def get_param_grid(dicts):
  return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

def model_fit_predict_CV(X, y, split, params):

    X_train, y_train = X.iloc[split[0],:], y.iloc[split[0], :]
    X_valid, y_valid   = X.iloc[split[1],:], y.iloc[split[1], :]
    
    reg = _doFitBoostedTreeRegressor(X_train, y_train, X_train.columns, params)
    y_pred = _doPredictBoostedTreeRegressor(X_valid, reg)
    
    df_test = X_valid

    # combine y_test and y_pred in 1 dataset
    df_test[y_test_name] = y_valid
    df_test[y_test_pred_name] = y_pred
    
    accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall = evaluate(df_test, actual=y_test_name, predicted=y_test_pred_name,
             thresh=0.8, rel_method='extremes', extr_type='high',
             coef=1.5, relevance_pts=None)
             
    return accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall

#make input data pipeline for tf model
def make_input_fn(X, y, n_epochs=None, shuffle=False):
    NUM_EXAMPLES = math.floor(len(y) / 2)

    def input_fn():
    
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        #print(dataset)
        if shuffle:
          dataset = dataset.shuffle(NUM_EXAMPLES)
        #For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES,  drop_remainder=False)
        #print(dataset)
        return dataset

    return input_fn

#make test input data pipeline for tf model
def make_input_fn_test(X, n_epochs=None, shuffle=False):
    NUM_EXAMPLES = math.floor(len(X) / 2)
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(dict(X))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES,drop_remainder=True)
        #print(dataset)
        return dataset

    return input_fn

#train the boosted tree
def _doFitBoostedTreeRegressor(X, Y, columns, params):
    # Define our feature columns
    fc = tf.feature_column
    feature_columns = []
    NUMERIC_COLUMNS = columns

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    # Creating the TF dataset
    train_input_fn = make_input_fn(X, Y)
    #print(train_input_fn, 'train input fn')

    # Defining the estimator (BoostedTreeRegressor)
    n_batches = 2
    est = tf.estimator.BoostedTreesRegressor(feature_columns, n_batches_per_layer=n_batches, **params)
    # Training the Model
    est.train(train_input_fn, max_steps=100)
    print("Done training for hyperparameter set" + str(params))
    return est

#predict on test values
def _doPredictBoostedTreeRegressor(X, reg):
    origShape = X.shape
    test_input_fn = make_input_fn_test(X)
    outData = reg.predict(test_input_fn)
    print(outData)

    out = []
    count = 1
    while (count <= origShape[0]):
        try:
            out.append(float(next(outData)['predictions']))
            count = count + 1
        except:
            break

    out = np.array(out)

    return out


def rarify_data(df, df_train, df_test, target_variable, method, extr_type, thresh, coef, control_pts):

    # get df_train and df_test
    # param df_train: the training data frame
    # param df_test: the testing data frame
    # param target_variable: name of the target variable column
    # return: df_train and df_test with equal class distribution between classes: rare and not rare

    print("checking null values in dataset when applying rarify")
    print(df.isnull().values.any())

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

    # number of rare values that are be in each of the train and test
    numraretrain = int(round(prare * len(df_train)))
    numraretest = int(round(prare * len(df_test)))

    print('number of rare in train: {}/{}'.format(numraretrain, len(df_train)))
    print('==> {}%%'.format((numraretrain/len(df_train))*100))
    print('number of rare in test: {}/{}'.format(numraretest, len(df_test)))
    print('==> {}%%'.format((numraretest / len(df_test))*100))

    rare_values = sorted(rare_values)

    # rare indices partitioned for each of the train and test
    rtrain = rare_values[:numraretrain]
    rtest = rare_values[numraretrain:]

    # get the relevance of each of the  dftrain and dftest
    yreltrain = [demandrel[d] for d in df_train[target_variable]]
    yreltest = [demandrel[d] for d in df_test[target_variable]]

    if len(rtrain) != numraretrain:
        raise ValueError('Incompatibility between the number of rare values that must be included in the '
                         'training data for equal class distribution and the obtained number of rare')

    if len(rtest) != numraretest:
        raise ValueError('Incompatibility between the number of rare values that must be included in the '
                         'testing data for equal class distribution and the obtained number of rare')

    return rtrain, rtest, yreltrain, yreltest, phi_params['control.pts'], loss_params, demandrel

#required error metrics
def error_metrics(y_test, y_pred, ncols):
    r2score, mase, rms, mse, re, pearson, pval, mae, adjusted_r2, spearman_corr, distance_corr, mape = calculate_errors(y_test, y_pred, ncols)
    print("The range for the output variable is:" + str(y_test.mean()))
    print("r2score : " + str(r2score))
    print("adj r2score : " + str(adjusted_r2))
    print("mae : " + str(mase))
    print("rmse : " + str(rms))
    print("mse : " + str(mse))
    print("re : " + str(re))
    print("pearson : " + str(pearson))
    print("spearman : " + str(spearman_corr))
    print("distance : " + str(distance_corr))
    print("mape : " + str(mape))

#evaluate ub error metrics
def evaluate(df, actual, predicted, thresh, rel_method='extremes', extr_type='high', coef=1.5, relevance_pts=None):
    y = np.array(df[actual])
    phi_params, loss_params, _ = get_phi_loss_params(y, rel_method, extr_type, coef, relevance_pts)

    nb_columns = len(list(df.columns.values)) - 1

    accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall = get_stats(df[actual], df[predicted], nb_columns, thresh, phi_params, loss_params)
    return accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall


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


def get_stats(y_test, y_pred, nb_columns, thr_rel, phi_params, loss_params):

    # Function to compute regression error metrics between actual and predicted values +
    # correlation between both using different methods: Pearson, Spearman, and Distance
    # param y_test: the actual values. Example df['actual'] (the string inside is the name
    # of the actual column. Example: df['LE (mm)'], df['demand'], etc.)
    # param y_pred: the predicted vlaues. Example df['predicted']
    # param nb_columns: number of columns <<discarding the target variable column>>
    # return: R2, Adj-R2, RMSE, MSE, MAE, MAPE

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if not isinstance(y_test, list):
        y_test = list(y_test)
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)

    n = len(y_test)

    r2_Score = r2_score(y_test, y_pred)  # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1)  # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
    mse_score = mean_squared_error(y_test, y_pred)  # MSE
    mae_score = mean_absolute_error(y_test, y_pred)  # MAE
    #print(np.asarray(np.abs(( np.array(y_test) - np.array(y_pred)) / np.array(y_test)), dtype=np.float64))
    mape_score = np.asarray(np.abs(( np.array(y_test) - np.array(y_pred)) / np.array(y_test)), dtype=np.float64).mean() * 100  # MAPE
    accuracy = 100 - mape_score
    aic = len(y_test) * np.log(mse_score)
    bic = len(y_test) * np.log(mse_score)
    nmi = normalized_mutual_info_score(y_test, y_pred)

    trues = np.array(y_test)
    preds = np.array(y_pred)

    method = phi_params['method']
    npts = phi_params['npts']
    controlpts = phi_params['control.pts']
    ymin = loss_params['ymin']
    ymax = loss_params['ymax']
    tloss = loss_params['tloss']
    epsilon = loss_params['epsilon']

    rmetrics = runit.eval_stats(trues, preds, thr_rel, method, npts, controlpts, ymin, ymax, tloss, epsilon)

    # create a dictionary of the r metrics extracted above
    rmetrics_dict = dict(zip(rmetrics.names, list(rmetrics)))

    if isinstance(y_pred[0], np.ndarray):
        y_pred_new = [x[0] for x in y_pred]
        y_pred = y_pred_new
    
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    distance_corr = distance.correlation(y_test, y_pred)

    print('\nUtility Based Metrics')
    print('F1: %.5f' % rmetrics_dict['ubaF1'][0])
    print('F2: %.5f' % rmetrics_dict['ubaF2'][0])
    print('F05: %.5f' % rmetrics_dict['ubaF05'][0])
    print('precision: %.5f' % rmetrics_dict['ubaprec'][0])
    print('recall: %.5f' % rmetrics_dict['ubarec'][0])

    print('\nRegression Error Metrics')
    print('R2: %.5f' % r2_Score)
    print('Adj-R2: %.5f' % adjusted_r2)
    print('RMSE: %.5f' % rmse_score)
    print('MSE: %.5f' % mse_score)
    print('MAE: %.5f' % mae_score)
    print('MAPE: %.5f' % mape_score)
    print('Accuracy: %.5f' % accuracy)
    print('aic: %.5f' % aic)
    print('bic: %.5f' % bic)
    print('nmi: %.5f' % nmi)

    print('\nCorrelations')
    print('Pearson: %.5f' % pearson_corr)
    print('Spearman: %.5f' % spearman_corr)
    print('Distance: %.5f' % distance_corr)
    return accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,rmetrics_dict['ubaF1'][0],rmetrics_dict['ubaF2'][0],rmetrics_dict['ubaF05'][0],rmetrics_dict['ubaprec'][0],rmetrics_dict['ubarec'][0]
    
    
def calculate_avg_error_metrics(mape_folds,  acc_folds, aic_folds, bic_folds, nmi_folds, d_f,sp_f, p_f, mae_f,mse_f, rmse_f, ar2_f, r2_f, f1_f, f2_f, f5_f,prec_f,recall_f,folds):
  print('\nUtility Based Metrics Across All', file=open(output_path +"output_CV_results.txt", "a"))
  avg_f1 = f1_f / folds
  print('F1: ' , avg_f1, file=open(output_path +"output_CV_results.txt", "a"))
  avg_f2 =  f2_f / folds
  print('F2: ' , avg_f2, file=open(output_path +"output_CV_results.txt", "a") )
  avg_f5 = f5_f / folds
  print('F05:' ,  avg_f5, file=open(output_path +"output_CV_results.txt", "a"))
  avg_prec = prec_f / folds
  print('precision: ', avg_prec  , file=open(output_path +"output_CV_results.txt", "a"))
  avg_recall = recall_f / folds
  print('recall:' , avg_recall , file=open(output_path +"output_CV_results.txt", "a"))
  
  print('\nRegression Error Metrics Across All', file=open(output_path +"output_CV_results.txt", "a"))
  avg_r2 = r2_f/folds
  print('R2:' , avg_r2, file=open(output_path +"output_CV_results.txt", "a"))
  avg_ar2 =  ar2_f/folds
  print('Adj-R2:' , avg_ar2, file=open(output_path +"output_CV_results.txt", "a"))
  avg_rmse = rmse_f/folds
  print('RMSE:' , avg_rmse , file=open(output_path +"output_CV_results.txt", "a"))
  avg_mse = mse_f/folds
  print('MSE:' , avg_mse , file=open(output_path +"output_CV_results.txt", "a"))
  avg_mae =  mae_f/folds
  print('MAE:' , avg_mae, file=open(output_path +"output_CV_results.txt", "a"))
  avg_mape = mape_folds/folds
  print('MAPE:' , avg_mape , file=open(output_path +"output_CV_results.txt", "a"))
  avg_acc = acc_folds/folds
  print('Accuracy:' , avg_acc , file=open(output_path +"output_CV_results.txt", "a"))
  avg_aic = aic_folds/folds
  print('AIC:' , avg_aic , file=open(output_path +"output_CV_results.txt", "a"))
  avg_bic = bic_folds/folds
  print('BIC:' , avg_bic , file=open(output_path +"output_CV_results.txt", "a"))
  avg_nmi = nmi_folds/folds
  print('NMI:' , avg_nmi , file=open(output_path +"output_CV_results.txt", "a"))


  print('\nCorrelations Across All', file=open(output_path +"output_CV_results.txt", "a"))
  avg_pearson =  p_f/folds
  print('Pearson:' , avg_pearson, file=open(output_path +"output_CV_results.txt", "a"))
  avg_spearman = sp_f/folds
  print('Spearman:' , avg_spearman , file=open(output_path +"output_CV_results.txt", "a"))
  avg_dist =  d_f/folds
  print('Distance:' , avg_dist, file=open(output_path +"output_CV_results.txt", "a"))
  return avg_f1, avg_f2, avg_f5, avg_prec, avg_recall, avg_r2, avg_ar2, avg_rmse, avg_mse, avg_mae, avg_mape, avg_acc, avg_aic, avg_bic, avg_nmi, avg_pearson, avg_spearman, avg_dist
  
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
    
def apply_smogn(df_train, smogn, target_variable, phi_params, thr_rel, Cperc, k, repl, dist, p, pert, plotdensity=False ):

    #method that applies SMOGN Algorithm to the current data frame
    # print('getting back values from oversampled R data frame')
    # print('before smogn')
    #print(pandas2ri.py2ri(df_train).head(), "this is py2ri")
    if smogn:
        smogned = runit.WFDIBS(
            fmla=get_formula(target_variable),
            dat= pandas2ri.py2ri(df_train),
            #dat=df_train,
            method=phi_params['method'][0],
            npts=phi_params['npts'][0],
            controlpts=phi_params['control.pts'],
            thrrel=thr_rel,
            Cperc=Cperc,
            k=k,
            repl=repl,
            dist=dist,
            p=p, 
            pert=pert)

        # print('after smogn')
        # print('before pandas2ri')
         #convert the oversampled R Data.Frame back to a pandas data frame
        smogned = pandas2ri.ri2py_dataframe(smogned)
        # print('after pandas2ri')

        if plotdensity:
            # density plot after smooting
            plot_density(smogned,target_variable,output_folder + 'plots/', 'density_after_smogn', 'Density Plot')

        X_train = np.array(smogned.loc[:, smogned.columns != target_variable])
        y_train = np.array(smogned.loc[:, target_variable])

        return X_train, y_train
  
def write_to_txt(filename, content):
  text_file = open(output_path + filename, "w")
  text_file.write(content)
  text_file.close()
  
def plot_actual_vs_predicted(df, predicted_variable):
  plt.plot(list(range(1, len(df) + 1)), df[y_test_name], color='b', label='actual')
  plt.plot(list(range(1, len(df) + 1)), df[predicted_variable], color='r', label='predicted')
  plt.legend(loc='best')
  plt.suptitle('actual vs. predicted')
  plt.savefig(output_path + 'actual_vs_predicted')
  plt.close()
    
def plot_actual_vs_predicted_scatter_bisector(df, predicted_variable):
  fig, ax = plt.subplots()
  ax.scatter(df[y_test_name], df[predicted_variable], c='black')
  lims = [
      np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
      np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
  ]
  ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
  ax.set_aspect('equal')
  ax.set_xlim(lims)
  ax.set_ylim(lims)
  plt.suptitle('actual vs. predicted forecasts')
  plt.savefig(output_path + 'actual_vs_predicted_scatter_plot')
  plt.close()


def plot_relevance(y, yrel, target_variable, output_folder, fig_name):
    reldict = {}
    y = y[target_variable]
    for i, e in enumerate(y):
        if e not in reldict:
            reldict[e] = yrel[i]

    reldict = dict(collections.OrderedDict(sorted(reldict.items())))
    plt.plot(list(reldict.keys()), list(reldict.values()))
    plt.xlabel(target_variable)
    plt.ylabel('relevance')

    plt.savefig(output_folder + fig_name)
    plt.close()

def plot_target_variable(df, df_resampled, output_column, output_folder, fig_name):
    y = df[output_column]
    y_resamp = df_resampled[output_column]
    plt.plot(list(range(len(y))), sorted(y), label = "original")
    plt.plot(list(range(len(y_resamp))), sorted(y_resamp), label = "resampled")
    plt.xlabel('Index')
    plt.ylabel(target_variable)
    plt.legend()
    plt.savefig(output_folder + fig_name)
    plt.close()
  
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


def split_train_test_valid(df, TRAIN_RATIO, TEST_RATIO):
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_train = pd.DataFrame()
    Y_test = pd.DataFrame()
    unique_sites = df["Site"].unique()
    print("Number of sites:", len(unique_sites))

    for site in unique_sites:
        df_site = df[df["Site"] == site]
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


def export_scores(scores, columnName):
    file_name = output_path + "score_sheet_final.csv"
    if not os.path.exists(file_name):
        print("path does not exist")
        df = pd.DataFrame(list())
        df.to_csv(file_name, index=False)
    else:
        print("path exists")
        df = pd.read_csv(file_name, delimiter=',')

    df["Error Metrics"] = scores.keys()
    df[columnName] = scores.values()
    df.to_csv(file_name, index=False)
    return df

def binary_encode_column(df, columnToEncode):
    encoder = ce.BinaryEncoder(cols=[columnToEncode])
    df_encoder = encoder.fit_transform(df[columnToEncode])
    df = pd.concat([df, df_encoder], axis=1)
    return df

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
df = df.drop(columnsToDrop, axis = 1)
#df = df[[output_column, 'Site', 'Eeflux_ET', 'Year']]

#df = df.drop(columnsToDrop, axis = 1)

#filter year 
df = df[df['Year'] > 1999]
df = df.drop(['Year'], axis = 1)
#filter EEflux
df.loc[(df['Eeflux_ET'] < 0.3) , 'Eeflux_ET'] = 0.3
df.loc[(df['Eeflux_ET'] >15), 'Eeflux_ET'] = 15

# set output variable between 1 and 15 only
#df = df[df['LE_bowen_corr_mm'].between(1, 15)]

df = binary_encode_column(df, "Site")
df = binary_encode_column(df, "Month")
df = binary_encode_column(df, "Vegetation")
df.drop(columns=['Month', 'Vegetation', 'Month_0', 'Vegetation_0', 'Site_0'], inplace=True)
print(df.columns)
print(df.shape)

lagsForColumns = [ "WS", "RH", "TA", "Eeflux_LST", "Eeflux_Albedo", "Eeflux_NDVI"]
df = generate_lags(df, lagsForColumns)
print(df.columns)

##drop nan for the first 5 rows of the generated lags only 5 rows will be removed in here
df.isnull().mean() * 10
df.dropna(inplace=True)
print(df.shape)

#df.drop(columnToDrop, axis = 1, inplace=True)
print("checking null values in the whole dataset")
#df.replace([np.inf, -np.inf], np.nan)
#df.dropna(inplace=True)
print(df.isnull().values.any())
print(df.columns)

#drop NaN values
#df.dropna(inplace=True)
#print(df)
# filter sites if ameri or euro
#df = df[df["Site Id"].str.startswith('US-')]
#df = df[~df["Site Id"].str.startswith('US-')]
#set output variable between 1 and 15 only

#df = df[df['LE_bowen_corr_mm'].between(1, 15)]
#
##df.loc[df['Eeflux_ET'] > 15, 'Eeflux_ET'] = 15
##df.loc[(df['Eeflux_ET'] < 0.3) & (df['Eeflux_ET'] > 0), 'Eeflux_ET'] = 0.3
#
##print(df['Eeflux_ET'].min())
##print(df['Eeflux_ET'].max())
#
##df.rename(columns_rename, inplace=True)
#print(df.shape)
#print(df.columns)
##drop desired columns, rename, and drop the nans
##print(df.shape)
##ET_cols = ['Eeflux_EToF','Eeflux_ETrF','Eeflux_ETo','Eeflux_ETr', 'Eeflux_ET']
##ET_cols = ['Eeflux_EToF','Eeflux_ETrF','Eeflux_ET','Eeflux_ETr']
##df = df.drop(ET_cols, axis = 1)
#
#df = binary_encode_column(df, "Site")
#df = binary_encode_column(df, "Month")
#df = binary_encode_column(df, "Vegetation")
#df.drop(columns=['Month', 'Vegetation', 'Month_0', 'Vegetation_0', 'Site_0'], inplace=True)
#print(df.columns)
#print(df.shape)
#
##df.dropna(inplace=True)
##print("Dropped columns")
#
##generate lags for columns
##lagsForColumns = [ "WS", "RH", "TA", "EEflux LST", "EEflux Albedo", "EEflux NDVI"]
#lagsForColumns = [ "WS", "RH", "TA", "Eeflux_LST", "Eeflux_Albedo", "Eeflux_NDVI"]
#df = generate_lags(df, lagsForColumns)
#print(df.columns)
#
##drop nan for the first 5 rows of the generated lags only 5 rows will be removed in here
#df.isnull().mean() * 10
#df.dropna(inplace=True)
#print(df.shape)
#
#
#
##df["residual"] = df["ET_bowen_corr(mm)"] - df["Eeflux_ET"]
##df["residual"] = (df["ET_bowen_corr(mm)"] - df["Eeflux_ET"])**2
##df["residual"] = df["ET_bowen_corr(mm)"] / df["Eeflux_ET"]
#
#
#
##df["residual"] = df["ETo"] - df["Eeflux_ETo"]
##df["residual"] = (df["ETo"] - df["Eeflux_ETo"])**2
##df["residual"] = df["ETo"] / df["Eeflux_ETo"]
#
##df['A'] = df['ET_bowen_corr_mm'] / df['ETo']
##df['B'] = df['Eeflux_ET'] / df['Eeflux_ETo']
#
#df['resid'] = df['Eeflux_ET'] / df['LE_bowen_corr_mm']
##df['resid'] = (df['A'] - df['B'])**2
#
##print(df)
#columnToDrop = ["Eeflux_ET", "LE_bowen_corr_mm"]
##columnToDrop = ["Eeflux_ETo", "ETo", "EToF", "ETrF", "Image_Id", "Eeflux_ET", "ET_bowen_corr_mm"]
#
##columnToDrop = ["Eeflux_ETo", "ETo", "Eeflux_ET", "ET_bowen_corr_mm"]
#df.drop(columnToDrop, axis = 1, inplace=True)
#print("checking null values in the whole dataset")
##df.replace([np.inf, -np.inf], np.nan)
##df.dropna(inplace=True)
#print(df.isnull().values.any())
##print(df["Eeflux_ET"].isnull().sum(axis=0), "number of nan et ")
#print(df.columns)
##df.dropna(inplace=True)
#
average_target_variable = np.average(df[output_column])

########################################################################################################################
                                            #Train and Test
########################################################################################################################
#split into train and test according to special split
X_train, X_test, y_train, y_test = split_train_test_valid(df, 0.8, 0.2)

#dropping site id after filtering the sites
columnToDrop = "Site"
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

#checking if null values exist here
print("checking null values in train")
print(df_train.isnull().values.any())
print("checking null values in test")
print(df_test.isnull().values.any())
df_train.to_csv(output_path + "df_train_before_rarify.csv")


if utility_based:
    #retrieve indexes of rare values to utilize in stratified folds
    df.dropna(inplace=True)
    rtrain, rtest, yreltrain, yreltest, phi_params, loss_params, targetrel = rarify_data(df, df_train, df_test, output_column, rel_method,extr_type, thr_rel,coef, relevance_pts)
    #df_train.to_csv(output_path + "df_train_after_rarify.csv")
    #plotting relevance values in both train and test target variables
    yreltrain = np.array(yreltrain).reshape(len(yreltrain), 1)
    plot_relevance(y_train, yreltrain, output_column, output_path, "relevance_values_train_data")
    plot_relevance(y_test, yreltest, output_column, output_path, "relevance_values_test_data")
    df_rel = df_train 
    df_rel["y_rel"] = yreltrain
    df_rel.to_csv(output_path+"yrel_df.csv")
    X_train = X_train.drop(["y_rel"], axis=1)

#Ensuring target variable column is dropped
X_train = X_train.drop([output_column], axis=1)
X_test = X_test.drop([output_column], axis=1)

#showing columns used in training and size of X_train
cols = X_train.columns
print("cols in X train after rarify are: ")
print(X_train.columns)
print("size of Xtrain is : ")
print(len(X_train))


########################################################################################################################
                                        #Scaling
########################################################################################################################

#if scaling: 
# if automatic:
#   #type desired col names in X to be scaled
#   all_columns = ['WS', 'WS-1', 'WS-2', 'WS-3', 'WS-4',
#      'WS-5', 'RH', 'RH-1', 'RH-2', 'RH-3', 'RH-4', 'RH-5', 'TA', 'TA-1',
#      'TA-2', 'TA-3', 'TA-4', 'TA-5', 'LE_bowen_corr_mm',
#      'Eeflux_LST', 'Eeflux_LST-1', 'Eeflux_LST-2', 'Eeflux_LST-3',
#      'Eeflux_LST-4', 'Eeflux_LST-5', 'Eeflux_NDVI', 'Eeflux_NDVI-1',
#      'Eeflux_NDVI-2', 'Eeflux_NDVI-3', 'Eeflux_NDVI-4', 'Eeflux_NDVI-5',
#      'Eeflux_Albedo', 'Eeflux_Albedo_1', 'Eeflux_Albedo-2',
#      'Eeflux_Albedo-3', 'Eeflux_Albedo-4', 'Eeflux_Albedo-5']
#   #standardize dataset
#   X_train = apply_scaling(X_train, all_columns, X_train)
#   X_test = apply_scaling(X_test, all_columns, X_train)
# else: 
#   df_scaled_manual = apply_scaling_manual(df, all_columns, X_train, scaling)

#all_columns = ['WS', 'WS-1', 'WS-2', 'WS-3', 'WS-4',
#  'WS-5', 'RH', 'RH-1', 'RH-2', 'RH-3', 'RH-4', 'RH-5', 'TA', 'TA-1',
#  'TA-2', 'TA-3', 'TA-4', 'TA-5', 'SW_IN', 'LE_bowen_corr_mm',
#  'Eeflux_LST', 'Eeflux_LST-1', 'Eeflux_LST-2', 'Eeflux_LST-3',
#  'Eeflux_LST-4', 'Eeflux_LST-5', 'Eeflux_NDVI', 'Eeflux_NDVI-1',
#  'Eeflux_NDVI-2', 'Eeflux_NDVI-3', 'Eeflux_NDVI-4', 'Eeflux_NDVI-5',
#  'Eeflux_Albedo', 'Eeflux_Albedo_1', 'Eeflux_Albedo-2',
#  'Eeflux_Albedo-3', 'Eeflux_Albedo-4', 'Eeflux_Albedo-5', 'SW_IN-1',
#  'SW_IN-2', 'SW_IN-3', 'SW_IN-4', 'SW_IN-5']
#X_train = Min_Max_auto(X_train, X_train, all_columns)
#X_test =  Min_Max_auto(X_train, X_test, all_columns)
#print(X_train)
    
# y_label = get_rare_idex(X_train,y_train,rtrain)
# y_train_label = np.array(y_train)
# for i in range(20):
#     if y_label[i] == 1:
#         print(y_label[i])
#         print(y_train_label[i])



########################################################################################################################
                                        #Random Search
########################################################################################################################
# define set of hyper-parameters
# params = {
# 'n_trees': [50, 150, 200, 250, 300, 350],
# 'max_depth': [1, 3, 5, 7, 9],
# 'learning_rate' :  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
# 'l1_regularization': [0, 0.001, 0.01, 0.1, 0.2, 0.3],
# 'l2_regularization' : [0, 0.001,  0.01, 0.1, 0.2, 0.3]
# }

# # set of hyper-parameters but with tree complexity and pruning
# params_with_complexity = {
# 'n_trees': [50, 150, 200, 250, 300],
# 'max_depth': [1, 3, 5, 7, 9],
# 'learning_rate' :  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
# 'l1_regularization': [0.01, 0.1, 0.2, 0.3], 
# 'l2_regularization' : [0.01, 0.1, 0.2, 0.3],
# 'tree_complexity' : [1,2,3],
# 'pruning_mode' : ['pre', 'post']
# }

if random_search:
    # do random search
    rng = np.random.RandomState(0)
    #specify random parameter number

    # getting list of hyper-parameters
    param_list = list(ParameterSampler(params_with_complexity, n_iter=n_params, random_state=rng))

    mape_list = []
    #looping over hyper-parameters
    for param in param_list:
        #fitting regressors
        start = time.time()
        reg = _doFitBoostedTreeRegressor(X_train, y_train, X_train.columns, param)
        y_pred = _doPredictBoostedTreeRegressor(X_test, reg)
        end = time.time()
        print("The time taken to train and predict is " + str(end - start) + " seconds")
        df_test[y_test_name_pred] = y_pred
        if utility_based:
            accuracy, aic, bic, nmi, mape = evaluate(df_test, actual=y_test_name, predicted=y_test_name_pred, thresh=0.8, rel_method='extremes', extr_type='high',coef=1.5, relevance_pts=None)
            mape_list.append(mape)
        else:
            error_metrics(y_test, y_pred)

    print("The best mape is " + str(min(mape_list)))
    index = mape_list.index(min(mape_list))
    print("The best hyper-params are " + str(param_list[index]))

########################################################################################################################
                                          #Grid Creation
########################################################################################################################

#The best hyper-params in random search were the following:
#params_best = {'n_trees': 100, 'max_depth': 5, 'learning_rate': 0.1, 'l1_regularization': 0, 'l2_regularization': 0}

#we shall do gridsearch around these hyper-params:
params_grid = {
'n_trees': [100],
'max_depth': [7,6],
'learning_rate' :  [0.1]
}

#create dict of hyper-params
if grid_search:
    grid = get_param_grid(params_grid)
    print("We will be trying " + str(len(grid)) +  " hyper-params" )

########################################################################################################################
                                        #Grid Search + CV
########################################################################################################################
if grid_search:
    if train_batches:
        mape_all, f1_all,f2_all,f5_all,prec_all,rec_all,r2_all,ar2_all,rmse_all,mse_all,mae_all,acc_all,aic_all,bic_all,nmi_all,pearson_all,spearman_all,dist_all = ([] for i in range(18))
        n_params = batch_size / (repetitions * folds)
        total_iter = len(grid) * folds * repetitions
        grid_start = ( batch_num - 1 )* n_params
        grid_end = batch_num * n_params
        grid_needed = grid[int(grid_start):int(grid_end)]
    else:
        mape_all, f1_all,f2_all,f5_all,prec_all,rec_all,r2_all,ar2_all,rmse_all,mse_all,mae_all,acc_all,aic_all,bic_all,nmi_all,pearson_all,spearman_all,dist_all = ([] for i in range(18))
        grid_needed = grid

    for param in grid_needed:
        acc_rep=aic_rep=bic_rep=nmi_rep=mape_rep= d_rep=sp_rep= p_rep= mae_rep=mse_rep= rmse_rep= ar2_rep= r2_rep= f1_rep= f2_rep= f5_rep=prec_rep=recall_rep=0

        for rep in range(repetitions):
            fold_indx = get_fold_indices(X_train,y_train,folds,rtrain)
            print("Calculated stratified fold indices and will start training")
            acc_folds = aic_folds = bic_folds = nmi_folds = mape_folds = d_f=sp_f= p_f= mae_f=mse_f= rmse_f= ar2_f= r2_f= f1_f= f2_f= f5_f=prec_f=recall_f = 0
            for fold in range(folds):
                print( " *************************Results for FOLD number " + str(fold) + "***************************** " )
                print("Columns used in X_train in CV are: ")
                print(X_train.columns)
                accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall = model_fit_predict_CV(X_train,y_train,fold_indx[fold], param)
                acc_folds += accuracy 
                aic_folds += aic
                bic_folds += bic
                nmi_folds += nmi
                mape_folds += mape_score
                d_f += distance_corr
                sp_f += spearman_corr
                p_f += pearson_corr
                mae_f += mae_score
                mse_f += mse_score
                rmse_f += rmse_score
                ar2_f += adjusted_r2
                r2_f += r2_Score
                f1_f += f1
                f2_f += f2 
                f5_f += f5 
                prec_f += prec
                recall_f += recall 
            print("For param " + str(param) , file=open(output_path +"output_CV_results.txt", "a") )
            print( " *************************FOLDS Average***************************** ", file=open(output_path +"output_CV_results.txt", "a") )
            avg_f1, avg_f2, avg_f5, avg_prec, avg_recall, avg_r2, avg_ar2, avg_rmse, avg_mse, avg_mae, avg_mape, avg_accuracy, avg_aic, avg_bic, avg_nmi, avg_pearson, avg_spearman, avg_dist = calculate_avg_error_metrics(mape_folds, acc_folds, aic_folds, bic_folds, nmi_folds,  d_f,sp_f, p_f, mae_f,mse_f, rmse_f, ar2_f, r2_f, f1_f, f2_f, f5_f,prec_f,recall_f,folds)
                    
            acc_rep += avg_accuracy 
            aic_rep += avg_aic
            bic_rep += avg_bic
            nmi_rep += avg_nmi
            mape_rep += avg_mape
            d_rep += avg_dist
            sp_rep += avg_spearman
            p_rep += avg_pearson
            mae_rep += avg_mae
            mse_rep += avg_mse
            rmse_rep += avg_rmse
            ar2_rep += avg_ar2
            r2_rep += avg_r2
            f1_rep += avg_f1
            f2_rep += avg_f2 
            f5_rep += avg_f5 
            prec_rep += avg_prec
            recall_rep += avg_recall
        
        print( "************************REPETITIONS Average*******************************", file=open(output_path +"output_CV_results.txt", "a") )
        avg_f1, avg_f2, avg_f5, avg_prec, avg_recall, avg_r2, avg_ar2, avg_rmse, avg_mse, avg_mae, avg_mape, avg_accuracy, avg_aic, avg_bic, avg_nmi, avg_pearson, avg_spearman, avg_dist = calculate_avg_error_metrics(mape_rep,acc_rep, aic_rep, bic_rep, nmi_rep, d_rep,sp_rep, p_rep, mae_rep,mse_rep, rmse_rep,ar2_rep, r2_rep, f1_rep, f2_rep, f5_rep,prec_rep,recall_rep, repetitions)
        
        print("average mape", avg_mape)
        mape_all.append(avg_mape)
        f1_all.append(avg_f1)
        f2_all.append(avg_f2)
        f5_all.append(avg_f5)
        prec_all.append(avg_prec)
        rec_all.append(avg_recall)
        r2_all.append(avg_r2)
        ar2_all.append(avg_ar2)
        rmse_all.append(avg_rmse)
        mse_all.append(avg_mse)
        mae_all.append(avg_mae)
        acc_all.append(avg_accuracy)
        aic_all.append(avg_aic)
        bic_all.append(avg_bic)
        nmi_all.append(avg_nmi)
        pearson_all.append(avg_pearson)
        spearman_all.append(avg_spearman)
        dist_all.append(avg_dist)

        

    print("The best mape is " + str(min(mape_all)))
    index = mape_all.index(min(mape_all))
    print(index)
    print(mape_all)
    print(len(grid_needed))
    print("The best hyper-params are " + str(grid_needed[index]))

    scores_dict = OrderedDict()
    scores_dict["mean target"] = average_target_variable  
    scores_dict["F1"] = f1_all[index]
    scores_dict["F2"] =  f2_all[index]
    scores_dict["F05"] = f5_all[index]
    scores_dict["Precision"] = prec_all[index]
    scores_dict["Recall"] = rec_all[index]
    scores_dict["R2"] = r2_all[index]
    scores_dict["Adjusted R2"] = ar2_all[index]
    scores_dict["RMSE"] = rmse_all[index]
    scores_dict["MSE"] = mse_all[index]
    scores_dict["MAE"] = mae_all[index]
    scores_dict["MAPE"] = mape_all[index]
    scores_dict["Accuracy"] = acc_all[index]
    scores_dict["Pearson C.C."] = pearson_all[index]
    scores_dict["Spearman C.C."] = spearman_all[index]
    scores_dict["Spatial Distance"] =  dist_all[index]
    scores_dict["NMI"] = nmi_all[index]
    scores_dict["AIC"] = aic_all[index]
    scores_dict["BIC"] = bic_all[index]
    scores_dict["Data Size train"] = '-'
    scores_dict["Data Size test"] = '-'
    scores_dict["Training Time (seconds)"] = '-'
    scores_dict["Testing Time (seconds)"] = '-'
    export_scores(scores_dict, "Validation Scores")

    write_to_txt('winning-hyperparams.txt', str(grid_needed[index]))
    winning_hyper = grid_needed[index]

else:
    #winning_hyper = {'n_trees': 100, 'max_depth': 7, 'learning_rate': 0.1}
    winning_hyper = {}
#######################################################################################################################
                                        #Applying Smogn
#######################################################################################################################

if smogn:
    #redefine my train data
    df_train = X_train
    df_train[target_variable] = y_train
    df_train.to_csv(output_path + "df_train_before_smogn_1.csv")
    #confirm if rel method matches rel points
    if rel_method == 'range' and relevance_pts is None:
        raise ValueError('You have set rel_method = range. You must provide relevance_pts as a matrix. Currently, it is None')
    #get phi loss params
    y_train_a = np.array(df_train[target_variable])
    phi_params, loss_params, relevance_values = get_phi_loss_params(y_train_a, rel_method, extr_type, coef,relevance_pts)
    df_train.to_csv(output_path + "df_train_before_smogn_after_phi.csv")
    
    #apply smogn on the df_train data 
    X_train,y_train = apply_smogn(df_train, smogn, target_variable, phi_params, thr_rel, Cperc, k, repl, dist, p, pert, plotdensity=False)
    print(X_train.shape)
    print(df_train.columns)
    print(df_train.shape)
    print(cols)
    #define X_train as df 
    X_train = pd.DataFrame(X_train, columns= cols)

    print("cols in X train after smogn are :" )
    print(X_train.columns)
    
    #define a smogned df_train
    df_train_smogned = X_train
    df_train_smogned[output_column] = y_train
    df_train_smogned.to_csv(output_path + 'df_train_smogned.csv')
########################################################################################################################
                                         #Fix one-hot encoded errors 
########################################################################################################################
X_train.to_csv(output_path + 'X_train_b4_1_hot.csv')
if smogn:
  if one_hot_encoded:
      count_abnormal(X_train)
      print("fixing one hot encoded cols")
      X_train = round_oversampled_one_hot_encoded(X_train) 
  
  else:
      print("there are no onehot encoded cols to be accounted for")


########################################################################################################################
                                         #Reporting Rarity Metrics 
########################################################################################################################

if smogn:
    print("The size of the original data is " + str(size_original))
    print("The size of the oversampled data is " + str(df_train_smogned.shape))
    yrelafter, distances = get_relevance_oversampling(df_train_smogned, output_column, targetrel)
    roversampled = get_rare_indices(df_train_smogned[output_column], yrelafter, thr_rel, phi_params['control.pts'])
    rare_train_after = (len(roversampled)/len(df_train_smogned)) * 100
    print("The percentage of rare values in dataset after smogn are "  + str(rare_train_after))

########################################################################################################################
                                        #Final Training
########################################################################################################################



if output_column in X_train:
    X_train = X_train.drop([output_column], axis=1)

print("Training model on the best hyper-params " + str(winning_hyper) )
print( " *************************Final Results on all Folds***************************** " )
print("Columns used in X_train in final training are: ")
print(X_train.columns)

#fitting regressor
df_test.to_csv(output_path + 'train_final_dataset.csv')
start_t = time.time()
reg = _doFitBoostedTreeRegressor(X_train, y_train, X_train.columns, winning_hyper)
end_t = time.time()
start_p = time.time()
y_pred = _doPredictBoostedTreeRegressor(X_test, reg)
end_p = time.time()

print("The time taken to train is " + str(end_t - start_t) + " seconds")
print("The time taken to test is " + str(end_p - start_p) + " seconds")

training_time = end_t - start_t
testing_time = end_p - start_p

#printing average target variable
print("The average target variable is " + str(y_test.mean()))
df_test = X_test
df_test[y_test_name] = y_test

# combine y_test and y_pred in 1 dataset
df_test[y_test_pred_name] = y_pred
df_test.to_csv(output_path + 'test_dataset.csv')

#plotting actual vs predicted
plot_actual_vs_predicted(df_test, y_test_pred_name)
plot_actual_vs_predicted_scatter_bisector(df_test, y_test_pred_name)

#plotting target variable is smogn is used
if smogn:
    print(df_train.columns, df_train_smogned.columns)
    plot_target_variable(df_train, df_train_smogned, output_column, output_path, 'target_variable')

#reporting error metrics
if utility_based:
    accuracy, aic, bic, nmi, mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall = evaluate(df_test, actual=y_test_name, predicted=y_test_pred_name,
            thresh=0.8, rel_method='extremes', extr_type='high',
            coef=1.5, relevance_pts=None)
else:
    nb_columns = len(list(df_test.columns.values)) - 1
    error_metrics(np.array(y_test), np.array(y_pred), nb_columns )


scores_dict = OrderedDict()
scores_dict["mean target"] = average_target_variable  
scores_dict["F1"] = f1
scores_dict["F2"] =  f2
scores_dict["F05"] = f5
scores_dict["Precision"] = prec
scores_dict["Recall"] = recall
scores_dict["R2"] = r2_Score
scores_dict["Adjusted R2"] = adjusted_r2
scores_dict["RMSE"] = rmse_score
scores_dict["MSE"] = mse_score
scores_dict["MAE"] = mae_score
scores_dict["MAPE"] = mape_score
scores_dict["Accuracy"] = accuracy
scores_dict["Pearson C.C."] = pearson_corr
scores_dict["Spearman C.C."] = spearman_corr
scores_dict["Spatial Distance"] =  distance_corr
scores_dict["NMI"] = nmi
scores_dict["AIC"] = aic
scores_dict["BIC"] = bic
scores_dict["Data Size train"] = len(df_train)
scores_dict["Data Size test"] = len(df_test)
scores_dict["Training Time (seconds)"] = training_time
scores_dict["Testing Time (seconds)"] = testing_time
export_scores(scores_dict, "Testing Scores")

#saving error metrics in file
with open(output_path + 'winning-model-scores.txt', 'a') as the_file:
     the_file.write('\nUtility Based Metrics'+'\n')
     the_file.write('F1: %.5f' % f1 + '\n')
     the_file.write('F2: %.5f' % f2+'\n')
     the_file.write('F05: %.5f' % f5+'\n')
     the_file.write('precision: %.5f' %prec+'\n')
     the_file.write('recall: %.5f' % recall+'\n')

     the_file.write('\nRegression Error Metrics'+'\n')
     the_file.write('R2: %.5f' % r2_Score+'\n')
     the_file.write('Adj-R2: %.5f' % adjusted_r2+'\n')
     the_file.write('RMSE: %.5f' % rmse_score+'\n')
     the_file.write('MSE: %.5f' % mse_score+'\n')
     the_file.write('MAE: %.5f' % mae_score+'\n')
     the_file.write('MAPE: %.5f' % mape_score+'\n')

     the_file.write('\nCorrelations'+'\n')
     the_file.write('Pearson: %.5f' % pearson_corr+'\n')
     the_file.write('Spearman: %.5f' % spearman_corr+'\n')
     the_file.write('Distance: %.5f' % distance_corr+'\n')

   
   




   
   




   
   


