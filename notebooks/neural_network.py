##Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import math

import tensorflow
print(tensorflow.__version__)
# from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow import feature_column
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
from math import log
import tensorflow.keras.backend as K
print("GPU Available: ", tensorflow.test.is_gpu_available())


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
from math import log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import re
import warnings
warnings.filterwarnings("ignore")


########################################################################################################################
                                            #Setting Global Variables
########################################################################################################################

#specify path of dataset
# user_path = "/home/sba19/scratch/smogn/"
user_path = "/home/yhh05/smogn/"
input_path = user_path + "All_Manual_Daily_Albedo_NDVI_LST_Cleaned.csv"
experiment_path = user_path + "Experiment Details.xlsx"


#specify saved file directory 
output_path = user_path + "output/"
output_result_path = output_path + "Daily/Experiment1/"

model_name = "all_manual_experiment"

#specify the output column
output_column = "LE_bowen_corr(mm)"

#specify name of target variable and name of predicted target variable
y_test_name = 'LE_bowen_corr(mm)'
y_test_pred_name = 'LE_bowen_corr_mm(pred)'

#specify one-hot encoded vector names
#For now we use month, vegetation and site id
one_hot_encoded = ['Site Id_1', 'Site Id_2', 'Site Id_3', 'Site Id_4', 'Site Id_5',
'Site Id_6', 'Month_1', 'Month_2', 'Month_3', 'Month_4','Vegetation_1', 'Vegetation_2','Vegetation_3']
                  
#specify desired split size
test_size = 0.2
  
#specify the option of utility based
utility_based = True

#specify if random search
random_search = False
#
#specify if you wish to apply over sampling by smogn
smogn = True

#smogn relate hyper-params
target_variable = "LE_bowen_corr(mm)"
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
thr_rel=0.8
Cperc="balance"
k=5
repl=False
dist="Euclidean"
p=2
pert=0.1


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
def calculate_errors(actual, predicted):
    r2score = r2_score(actual, predicted)
    mase = mean_absolute_error(actual, predicted)
    rms = sqrt(mean_squared_error(actual, predicted))
    mse = mean_squared_error(actual, predicted)
    re = (mse / np.mean(predicted)) * 100
    pearson, pval = stats.pearsonr(actual.ravel(), predicted.ravel())
    mae = np.mean(np.abs((actual - predicted) / actual)) * 100
    return r2score, mase, rms, mse, re, pearson, pval, mae

#get indices of folds in Stratified KFold CV
def get_fold_indices(X,y,n_splits,rare_values):
    rare_vec = [1 if i in rare_values else 0 for i in range(len(y))]
    y = np.array(rare_vec)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=123)
    folds = list(splitter.split(X, y))
    return folds
  
#get grid of all hyper-parameters
def get_param_grid(dicts):
  return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]


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
    rare_values, phi_params, loss_params, yrel = get_rare(y, method, extr_type, thresh, coef, control_pts)

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

    print('number of rare that must be in train: {}/{}'.format(numraretrain, len(df_train)))
    print('==> {}%%'.format((numraretrain/len(df_train))*100))
    print('number of rare that must be in test: {}/{}'.format(numraretest, len(df_test)))
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

    return df_train, df_test, rtrain, rtest, yreltrain, yreltest, phi_params['control.pts'], loss_params, demandrel

#required error metrics
def error_metrics(y_test, y_pred):
    r2score, mase, rms, mse, re, pearson, pval, mae = calculate_errors(y_test, y_pred)
    print("The range for the output variable is:" + str(y_test.mean()))
    print("r2score : " + str(r2score))
    print("mae : " + str(mase))
    print("rmse : " + str(rms))
    print("mse : " + str(mse))
    print("re : " + str(re))
    print("pearson : " + str(pearson))
    print("mape : " + str(mae))

#evaluate ub error metrics
def evaluate(df, actual, predicted, num_params, thresh, rel_method='extremes', extr_type='high', coef=1.5, relevance_pts=None):
    y = np.array(df[actual])
    phi_params, loss_params, _ = get_phi_loss_params(y, rel_method, extr_type, coef, relevance_pts)

    nb_columns = len(list(df.columns.values)) - 1

    mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall, accuracy, aic, bic = get_stats(df[actual], df[predicted], nb_columns, thresh, phi_params, loss_params, num_params)
    return mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall, accuracy, aic, bic


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


def get_stats(y_test, y_pred, nb_columns, thr_rel, phi_params, loss_params, num_params):

    # Function to compute regression error metrics between actual and predicted values +
    # correlation between both using different methods: Pearson, Spearman, and Distance
    # param y_test: the actual values. Example df['actual'] (the string inside is the name
    # of the actual column. Example: df['LE (mm)'], df['demand'], etc.)
    # param y_pred: the predicted vlaues. Example df['predicted']
    # param nb_columns: number of columns <<discarding the target variable column>>
    # return: R2, Adj-R2, RMSE, MSE, MAE, MAPE
    
    #convert to float for it to work for AIC and BIC data should all be of the same format i.e float not mix of types
    y_test_f =  [float(item) for item in y_test.values]
    y_predict_f =  [float(item) for item in y_pred]

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
    accuracy_score = 100 - np.mean(mape_score) # Accuracy
    
    def calculate_aic(n, mse, num_params):
        aic = n * log(mse) + 2 * num_params
        return aic

    def calculate_bic(n, mse, num_params):
        bic = n * log(mse) + num_params * log(n)
        return bic

    aic = str(round(calculate_aic(len(y_test), mse_score, num_params), 2))
    bic = str(round(calculate_bic(len(y_test), mse_score, num_params), 2))

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
    print('F1: %.2f' % rmetrics_dict['ubaF1'][0])
    print('F2: %.2f' % rmetrics_dict['ubaF2'][0])
    print('F05: %.2f' % rmetrics_dict['ubaF05'][0])
    print('precision: %.2f' % rmetrics_dict['ubaprec'][0])
    print('recall: %.2f' % rmetrics_dict['ubarec'][0])

    print('\nRegression Error Metrics')
    print('R2: %.2f' % r2_Score)
    print('Adj-R2: %.2f' % adjusted_r2)
    print('RMSE: %.2f' % rmse_score)
    print('MSE: %.2f' % mse_score)
    print('MAE: %.2f' % mae_score)
    print('MAPE: %.2f' % mape_score)
    print('Accuracy: %.2f' % accuracy_score)
    print('AIC:', aic)
    print('BIC:',  bic)

    print('\nCorrelations')
    print('Pearson: %.2f' % pearson_corr)
    print('Spearman: %.2f' % spearman_corr)
    print('Distance: %.2f' % distance_corr)
    return mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,rmetrics_dict['ubaF1'][0],rmetrics_dict['ubaF2'][0],rmetrics_dict['ubaF05'][0],rmetrics_dict['ubaprec'][0],rmetrics_dict['ubarec'][0], accuracy_score, aic, bic
    
    
def calculate_avg_error_metrics(mape_folds, d_f,sp_f, p_f, mae_f,mse_f, rmse_f, ar2_f, r2_f, f1_f, f2_f, f5_f,prec_f,recall_f,folds, acc_f, aic_f, bic_f):
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
  
  avg_acc = acc_f/folds
  print('ACCURACY:' , avg_acc , file=open(output_path +"output_CV_results.txt", "a"))
  avg_aic = aic_f/folds
  print('AIC:' , avg_aic , file=open(output_path +"output_CV_results.txt", "a"))
  avg_bic = bic_f/folds
  print('BIC:' , avg_bic , file=open(output_path +"output_CV_results.txt", "a"))
  
  print('\nCorrelations Across All', file=open(output_path +"output_CV_results.txt", "a"))
  avg_pearson =  p_f/folds
  print('Pearson:' , avg_pearson, file=open(output_path +"output_CV_results.txt", "a"))
  avg_spearman = sp_f/folds
  print('Spearman:' , avg_spearman , file=open(output_path +"output_CV_results.txt", "a"))
  avg_dist =  d_f/folds
  print('Distance:' , avg_dist, file=open(output_path +"output_CV_results.txt", "a"))
  return avg_f1, avg_f2, avg_f5, avg_prec, avg_recall, avg_r2, avg_ar2, avg_rmse, avg_mse, avg_mae, avg_mape, avg_pearson, avg_spearman, avg_dist, avg_acc, avg_aic, avg_bic
  
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
    # print('zamatet')
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
    
    unique_sites = df["Site Id"].unique()
    print("Number of sites:", len(unique_sites))

    for site in unique_sites:
        df_site = df[df["Site Id"] == site]
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


def write_dict_to_json(path, content):
    with open(path + '.json', 'w') as file:
        file.write(json.dumps(content, ensure_ascii=False))


########################################################################################################################
                                            #Establish a connection to R library
########################################################################################################################
rpy2.robjects.numpy2ri.activate()
runit = robjects.r
runit['source']('/home/sba19/scratch/smogn/smogn.R')

########################################################################################################################
                                            #Read and Preprocess Dataset
########################################################################################################################


## Get Data
# df_experiment = pd.read_excel(experiment_path, "Daily")
# df_experiment

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
columnsToDrop = ['Date','Year','Month','Day',
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
                 'ETo', 'EToF', 'ETr', 'ETrF']
df = df.drop(columnsToDrop, axis = 1)
df.dropna(inplace=True)

#generate lags for columns
lagsForColumns = ["SW_IN", "WS", "RH", "TA", "EEflux LST", "EEflux Albedo", "EEflux NDVI"]
df = generate_lags(df, lagsForColumns)
print(df.columns)

#drop nan for the first 5 rows of the generated lags only 5 rows will be removed in here
df.isnull().mean() * 10
df.dropna(inplace=True)
print(df.shape)

print("checking null values in the whole dataset")
print(df.isnull().values.any())


num_features = df.shape[1]
print(num_features)


########################################################################################################################
                                            #Train and Test
########################################################################################################################

#split into train and test according to special split
X_train, X_test, y_train, y_test = split_train_test_valid(df, 0.8, 0.2)

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

#checking if null values exist here
print("checking null values in train")
print(df_train.isnull().values.any())
print("checking null values in test")
print(df_test.isnull().values.any())

#retrieve indexes of rare values to utilize in stratified folds
df_train_rare, df_test_rare, rtrain, rtest, yreltrain, yreltest, phi_params, loss_params, targetrel = rarify_data(df, df_train, df_test, output_column, rel_method, extr_type, thr_rel,coef, relevance_pts)

#plotting relevance values in both train and test target variables
yreltrain = np.array(yreltrain).reshape(len(yreltrain), 1)
plot_relevance(y_train, yreltrain, output_column, output_path, "relevance_values_train_data")
plot_relevance(y_test, yreltest, output_column, output_path, "relevance_values_test_data")

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

#if not automatic specify desired column names
all_columns = X_train.columns
#specify the scaling type for each column
scaling = ['MinMax', 'Standard', 'Robust', 'MinMax + PowerTransform', 'Standard + PowerTransform']

#specify if scaling
scaling = True

#specify if automatic or manual scaling
automatic = True

if scaling: 
   if automatic:
       #type desired col names in X to be scaled
       all_columns = list(X_train.columns)
       #standardize dataset
       #TODO: sawad print scaling used
       X_train = apply_scaling(X_train, all_columns, X_train)
       X_test = apply_scaling(X_test, all_columns, X_train)
   else: 
        df_scaled_manual = apply_scaling_manual(df, all_columns, X_train, scaling)

#####################################################################################################################################################################################################################################################


########################################################################################################################
                                          #Grid Creation
########################################################################################################################

#specify if grid search
grid_search = True

#spcify repetitions and folds for repeated stratified cross validation
repetitions = 1
folds = 5

#we shall do gridsearch around these hyper-params:
params_grid = {
 'first_neuron':[64],
 'activation' : ['softmax'],
 'init': ['uniform'],
 'dropout_rate' : [0.4],
 'dense_layer_sizes' : [2],
 'optimizer' : ['Adam'],
 'loss': ['mean_absolute_percentage_error', 'mse'],
 'epochs':[10],
 'batch_size':[64]
}
#create dict of hyper-params
if grid_search:
    grid_needed = get_param_grid(params_grid)
    print("We will be trying " + str(len(grid_needed)) +  " hyper-params" )
    
else:
    grid_needed = {
    'first_neuron':64,
     'activation' : 'softmax',
     'init': 'uniform',
     'dropout_rate' : 0.4,
     'dense_layer_sizes' : 2,
     'optimizer' : 'Adam',
     'loss': 'mse',
     'epochs':500,
     'batch_size':64
    }
########################################################################################################################
                                        #Grid Search + CV
########################################################################################################################

##Create model and start training

n_input = X_train.shape[1]
n_classes = 1
print("num of input:", n_input, "num of classes:", n_classes)


def create_model(first_neuron=30,
                 activation='relu',
                 init='uniform',
                 dropout_rate = 0.0,
                 dense_layer_sizes = 2,
                 optimizer='Adam',
                 loss='mean_absolute_percentage_error'):
    # Create model
    model = tensorflow.keras.Sequential()
    model.add(Dense(first_neuron, input_dim=n_input, kernel_initializer=init, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    for layer_size in range(dense_layer_sizes):
        model.add(Dense(first_neuron, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="linear"))
    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae', 'accuracy', 'mape'])
    return model

create_model().summary


model = KerasRegressor(build_fn=create_model)


# Prepare the Grid
param_grid = {
              'first_neuron':[64],
              'activation' : ['softmax'],
              'init': ['uniform'],
              'dropout_rate' : [0.4],
              'dense_layer_sizes' : [2],
              'optimizer' : ['Adam'],
              'loss': ['mse'],
              'epochs':[500],
              'batch_size':[64]
             }

best_params = param_grid


def model_fit_predict_CV(X, y, split, params):
    X_train, y_train = X.iloc[split[0],:], y.iloc[split[0], :]
    X_valid, y_valid   = X.iloc[split[1],:], y.iloc[split[1], :]
    
    model = create_model(params["first_neuron"],
                        params["activation"],
                        params["init"],
                        params["dropout_rate"],
                        params["dense_layer_sizes"],
                        params["optimizer"],
                        params["loss"])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)
    mc = ModelCheckpoint(output_path + model_name, monitor='val_mape', mode='min', verbose=0, save_best_only=True)
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_valid, y_valid),
                        epochs=params["epochs"],
                       verbose=0,
                       callbacks=[es, mc])
                       
                       
    ## Test data
    y_pred = model.predict(X_valid)
    
    df_test = X_valid

    # combine y_test and y_pred in 1 dataset
    df_test[y_test_name] = y_valid
    df_test[y_test_pred_name] = y_pred

    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    print("num parameters:", trainable_count)
    num_params = trainable_count
    
    mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall, acc, aic, bic = evaluate(df_test, actual=y_test_name, predicted=y_test_pred_name, num_params=num_params, thresh=0.8, rel_method='extremes', extr_type='high', coef=1.5, relevance_pts=None)
             
    return mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall, acc, aic, bic


if grid_search:
    for param in grid_needed:
        mape_all = []
        acc_rep=aic_rep=bic_rep=mape_rep= d_rep=sp_rep= p_rep= mae_rep=mse_rep= rmse_rep= ar2_rep= r2_rep= f1_rep= f2_rep= f5_rep=prec_rep=recall_rep=0
        for rep in range(repetitions):
            fold_indx = get_fold_indices(X_train,y_train,folds,rtrain)
            print("Calculated stratified fold indices and will start training")
            acc_f=aic_f=bic_f=mape_folds = d_f=sp_f= p_f= mae_f=mse_f= rmse_f= ar2_f= r2_f= f1_f= f2_f= f5_f=prec_f=recall_f = 0
            for fold in range(folds):
                print( " *************************Results for FOLD number " + str(fold) + "***************************** " )
                print("Columns used in X_train in CV are: ")
                print(X_train.columns)
                mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall, acc, aic, bic = model_fit_predict_CV(X_train,y_train,fold_indx[fold], param)
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
                acc_f += acc
                aic_f += float(aic)
                bic_f += float(bic)
            print("For param " + str(param) , file=open(output_path +"output_CV_results.txt", "a") )
            print( " *************************FOLDS Average***************************** ", file=open(output_path +"output_CV_results.txt", "a") )
            avg_f1, avg_f2, avg_f5, avg_prec, avg_recall, avg_r2, avg_ar2, avg_rmse, avg_mse, avg_mae, avg_mape, avg_pearson, avg_spearman, avg_dist, avg_acc, avg_aic, avg_bic = calculate_avg_error_metrics(mape_folds, d_f,sp_f, p_f, mae_f,mse_f, rmse_f, ar2_f, r2_f, f1_f, f2_f, f5_f,prec_f,recall_f,folds, acc_f, aic_f, bic_f)
                    
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
            acc_rep += avg_acc
            aic_rep += float(avg_aic)
            bic_rep += float(avg_bic)
        
        print( "************************REPETITIONS Average*******************************", file=open(output_path +"output_CV_results.txt", "a") )
        avg_f1, avg_f2, avg_f5, avg_prec, avg_recall, avg_r2, avg_ar2, avg_rmse, avg_mse, avg_mae, avg_mape, avg_pearson, avg_spearman, avg_dist, avg_acc, avg_aic, avg_bic = calculate_avg_error_metrics(mape_rep, d_rep,sp_rep, p_rep, mae_rep,mse_rep, rmse_rep,ar2_rep, r2_rep, f1_rep, f2_rep, f5_rep,prec_rep,recall_rep, repetitions, acc_rep, aic_rep, bic_rep)
                
        mape_all.append(avg_mape)

    print("The best mape is " + str(min(mape_all)))
    index = mape_all.index(min(mape_all))
    print("The best hyper-params are " + str(grid_needed[index]))

    write_to_txt('winning-hyperparams.txt', str(grid_needed[index]))
    winning_hyper = grid_needed[index]

else:
    winning_hyper = grid_needed
    

#######################################################################################################################
                                        #Applying Smogn
#######################################################################################################################

if smogn:
    #redefine my train data
    df_train = X_train
    df_train[target_variable] = y_train

    #confirm if rel method matches rel points
    if rel_method == 'range' and relevance_pts is None:
        raise ValueError('You have set rel_method = range. You must provide relevance_pts as a matrix. Currently, it is None')

    #get phi loss params
    y_train_a = np.array(df_train[target_variable])
    phi_params, loss_params, relevance_values = get_phi_loss_params(y_train_a, rel_method, extr_type, coef,relevance_pts)

    #apply smogn on the df_train data
    X_train, y_train = apply_smogn(df_train, smogn, target_variable, phi_params, thr_rel, Cperc, k, repl, dist, p, pert, plotdensity=False)

    #define X_train as df
    X_train = pd.DataFrame(X_train, columns= cols)

    print("cols in X train after smogn are :" )
    print(X_train.columns)
    
    #define a smogned df_train
    df_train_smogned = X_train
    df_train_smogned[output_column] = y_train


########################################################################################################################
                                         #Fix one-hot encoded errors
########################################################################################################################

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

start = time.time()

#TODO: sara split X_train into train and validataion

if output_column in X_train:
    X_train = X_train.drop([output_column], axis=1)


#Split X_train into X_train and X_valid
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Y_train shape:", y_train.shape, "y_test shape:", y_test.shape, "y_valid shape:", y_valid.shape)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

n_input = X_train.shape[1]
n_classes = 1
print("num of input:", n_input, "num of classes:", n_classes)


print("Training model on the best hyper-params " + str(winning_hyper) )
print( " *************************Final Results on all Folds***************************** " )
print("Columns used in X_train in final training are: ")
print(X_train.columns)

#fitting regressor
model = create_model(winning_hyper["first_neuron"],
                    winning_hyper["activation"],
                    winning_hyper["init"],
                    winning_hyper["dropout_rate"],
                    winning_hyper["dense_layer_sizes"],
                    winning_hyper["optimizer"],
                    winning_hyper["loss"])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)
mc = ModelCheckpoint(output_path + model_name, monitor='val_mape', mode='min', verbose=0, save_best_only=True)
history = model.fit(X_train_scaled,
                    y_train,
                    validation_data=(X_valid_scaled, y_valid),
                    epochs=300,
                   verbose=0,
                   callbacks=[es, mc])

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
print("num parameters:", trainable_count)
num_params = trainable_count


## Plot learning curve
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE Loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_path + 'learning_curve_train_val_mse')
plt.close()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE Loss')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_path +  'learning_curve_train_val_mae')
plt.close()

plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('MAPE Loss')
plt.ylabel('mape')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_path +  'learning_curve_train_val_mape')
plt.close()

## Evaluate scores for train, test and validation
loss, train_mse, train_mae, train_acc, train_mape = model.evaluate(X_train_scaled, y_train, verbose=2)
loss, val_mse, val_mae, val_acc, val_mape = model.evaluate(X_valid_scaled, y_valid, verbose=2)
loss, test_mse, test_mae, test_acc, test_mape = model.evaluate(X_test, y_test, verbose=2)

print("train mse:", train_mse, "validation mse:", val_mse, "test mse:", test_mse)
print("train mae:", train_mae, "validation mae:", val_mae, "test mae:", test_mae)
print("train mape:", train_mape, "validation mape:", val_mape, "test mape:", test_mape)

validation_scores = {
    "MAE": str(round(val_mae, 2)),
    "MSE": str(round(val_mse, 2)),
    "MAPE": str(round(val_mape, 2)),
    "Accuracy": str(round(100 - np.mean(val_mape), 2))
}

train_scores = {
    "MAE": str(round(train_mae, 2)),
    "MSE": str(round(train_mse, 2)),
    "MAPE": str(round(train_mape, 2)),
    "Accuracy": str(round(100 - np.mean(train_mape), 2))
}

test_scores = {
    "MAE": str(round(test_mae, 2)),
    "MSE": str(round(test_mse, 2)),
    "MAPE": str(round(test_mape, 2)),
    "Accuracy": str(round(100 - np.mean(test_mape), 2))
}

scores = {
    "Test Scores": test_scores,
    "Validation Scores": validation_scores,
    "Train Scores": train_scores
}
write_dict_to_json(output_path + "evaluation_scores", scores)

                    
## Test data
y_pred = model.predict(X_test)
end = time.time()
print("The time taken to train and predict is " + str(end - start) + " seconds")

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
    plot_target_variable(df, df_train_smogned, output_column, output_path, 'target_variable')

#reporting error metrics
if utility_based:
   mape_score,distance_corr,spearman_corr,pearson_corr,mae_score,mse_score,rmse_score,adjusted_r2,r2_Score,f1,f2,f5,prec,recall, acc, aic, bic  = evaluate(df_test, actual=y_test_name, predicted=y_test_pred_name, num_params=num_params, thresh=0.8, rel_method='extremes', extr_type='high', coef=1.5, relevance_pts=None)
else:
    error_metrics(y_test, y_pred)

