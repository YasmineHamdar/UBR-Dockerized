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
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import RobustScaler,OneHotEncoder,MinMaxScaler,PowerTransformer,StandardScaler
from scipy.stats import normaltest
from numpy.random import randn
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from numpy import *
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats.stats import pearsonr, spearmanr
from scipy import stats
import re
import warnings
warnings.filterwarnings("ignore")

#checks if the input is gaussian by shapiro wilk test
def check_distribution_shapiro(col):
    stat, p = shapiro(col)
    alpha = 0.05
    if p > alpha:
        gaussian = True
    else:
        gaussian = False
        
    return gaussian


def check_distribution_dagostino(col):
    stat, p = normaltest(col)
    alpha = 0.05
    if p > alpha:
        gaussian = True
    else:
        gaussian = False
        
    return gaussian

#splits data into train and test
def train_test_splitting(X,y,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

#does suitable standardization
def standardization_needed(col, X):
    col = col.values.reshape(-1, 1)
    X = X.values.reshape(-1, 1)
    #checks if column has zero values
    if 0 not in col:
        col_trans = Standard(col,X)
        col_trans_pow = apply_power_trans(col_trans)
    else:
        #if there are zero values, applies MinMaxScaler 
        #check range of values
        col_trans = Min_Max(col,X)
        col_trans_pow = apply_power_trans(col_trans)          
    return  np.ravel(col_trans_pow)

def standardization_needed_manual(col, X, scaling):
    col = col.values.reshape(-1, 1)
    X = X.values.reshape(-1, 1)
    if scaling == 'None':
        col_trans_final = col
    if scaling == 'MinMax' :
        col_trans_final = Min_Max(col,X)
    if scaling == 'Standard' :
        col_trans_final = Standard(col,X)
    if scaling == 'Robust':
        col_trans_final = Robust(col,X)
    if scaling == 'MinMax + PowerTransform':
        col_trans = Min_Max(col,X)
        col_trans_final = apply_power_trans(col_trans)
    if scaling == 'Standard + PowerTransform':
        col_trans = Standard(col,X)
        col_trans_final = apply_power_trans(col_trans)
    return np.ravel(col_trans_final)
        
        
def Min_Max(col,X):
    if any(n < 0 for n in col):
        scaler = MinMaxScaler((-1,1))
    else: 
        scaler = MinMaxScaler((0,1))
    scaler.fit(X)
    col_trans = scaler.transform(col)
    return col_trans

def Standard(col,X):
    if 0 not in col:
        #if no zero values, apply StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)
        col_trans = scaler.transform(col)
        return col_trans
    else: 
        print('column has 0 values, cannot apply standard scaling')
        return col
    
def Robust(col,X):
    scaler = RobustScaler()
    scaler.fit(X)
    col_trans = scaler.transform(col)
    return col_trans
    

def apply_power_trans(col_trans):
    if any(n <= 0 for n in col_trans):
    #if there are negative or zero values, applies yeo-johnson transform
        pt = PowerTransformer('yeo-johnson')
        col_trans_pow = pt.fit_transform(col_trans)
        #checks if column values are strictly positive
    elif all(n > 0 for n in col):
        #if values are strictly positive, applies box-cox transform
        pt = PowerTransformer('box-cox')
        col_trans_pow = pt.fit_transform(col_trans)
    return col_trans_pow

def apply_scaling(df, columns, X_train): 
    df_scaled = pd.DataFrame(columns=columns)
    for i in columns: 
        #checking if data is Gaussian
        if not check_distribution_shapiro(df[i]) and not check_distribution_dagostino(df[i]):
            print(str(i) + ' does not have a Gaussian distribution and will be scaled')
            #scaling data
            df_scaled[i] = standardization_needed(df[i], X_train[i])
        else: 
            df_scaled[i] = df[i]
            print(str(i) + ' has a gaussian distribution')
    return df_scaled

def apply_scaling_manual(df, columns, X_train, scaling):
    iterator = 0
    if len(scaling) != len(columns):
        print("Please specify scaling for all columns listed")
        return
    else:
        df_scaled = pd.DataFrame(columns=columns)
        for i in columns: 
            #checking if data is Gaussian
            if not check_distribution_shapiro(df[i]) and not check_distribution_dagostino(df[i]):
                print(str(i) + ' does not have a Gaussian distribution and will be scaled')
                #scaling data
                df_scaled[i] = standardization_needed_manual(df[i], X_train[i], scaling[iterator])
                iterator = iterator + 1
            else: 
                df_scaled[i] = df[i]
                print(str(i) + ' has a gaussian distribution')
        return df_scaled

def calculate_errors(actual,predicted):
    r2score = r2_score(actual,predicted)
    mase = mean_absolute_error(actual,predicted)
    rms = sqrt(mean_squared_error(actual,predicted))
    mse = mean_squared_error(actual,predicted)
    re = (mse / np.mean(predicted)) * 100
    pearson, pval = stats.pearsonr(actual.ravel(),predicted.ravel())
    mae = np.mean(np.abs((actual - predicted) / actual)) * 100
    return r2score,mase,rms,mse,re,pearson,pval,mae

def fit_predict(X_train_scaled,X_test_scaled,y_train):
    model = SVR()
    model.fit(X_train_scaled,y_train)
    pred_y = model.predict(X_test_scaled)
    return pred_y
    
def error_metrics(y_test,y_pred):
    r2score,mase,rms,mse,re,pearson,pval,mae = calculate_errors(y_test,y_pred)
    print("The range for the output variable is:" + str(y_test.mean()))
    print("r2score : " + str(r2score))
    print("mae : " + str(mase))
    print("rmse : " + str(rms))
    print("mse : " + str(mse))
    print("re : " + str(re))
    print("pearson : " + str(pearson))
    print("mape : " + str(mae))
    
def evaluate(df, actual, predicted, thresh, rel_method='extremes', extr_type='high', coef=1.5, relevance_pts=None):
    y = np.array(df[actual])
    phi_params, loss_params, _ = get_phi_loss_params(y, rel_method, extr_type, coef, relevance_pts)

    nb_columns = len(list(df.columns.values)) - 1

    get_stats(df[actual], df[predicted], nb_columns, thresh, phi_params, loss_params)


def get_phi_loss_params(y, rel_method, extr_type='high', coef=1.5, relevance_pts=None):
    '''
    get the parameters of the relevance function
    :param df: dataframe being used
    :param target_variable: name of the target variable
    :param rel_method: either 'extremes' or 'range'
    :param extr_type: either 'high', 'low', or 'both' (defualt)
    :param coef: default: 1.5
    :param relevance_pts: the relevance matrix in case rel_method = 'range'
    :return: phi parameters and loss parameters
    '''

    if relevance_pts is None:
        print('WILL NOT USE RELEVANCE MATRIX')
        params = runit.get_relevance_params_extremes(y, rel_method=rel_method, extr_type=extr_type, coef=coef)
    else:
        print('USING RELEVANCE MATRIX')
        params = runit.get_relevance_params_range(y, rel_method=rel_method, extr_type=extr_type, coef=coef, relevance_pts=relevance_pts)

    # phi params and loss params
    phi_params = params[0]
    loss_params = params[1]
    relevance_values = params[2]

    phi_params = dict(zip(phi_params.names, list(phi_params)))
    loss_params = dict(zip(loss_params.names, list(loss_params)))

    return phi_params, loss_params, relevance_values


def get_stats(y_test, y_pred, nb_columns, thr_rel, phi_params, loss_params):
    '''
    Function to compute regression error metrics between actual and predicted values +
    correlation between both using different methods: Pearson, Spearman, and Distance
    :param y_test: the actual values. Example df['actual'] (the string inside is the name
    of the actual column. Example: df['LE (mm)'], df['demand'], etc.)
    :param y_pred: the predicted vlaues. Example df['predicted']
    :param nb_columns: number of columns <<discarding the target variable column>>
    :return: R2, Adj-R2, RMSE, MSE, MAE, MAPE
    '''

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if not isinstance(y_test, list):
        y_test = list(y_test)
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)

    n = len(y_test)

    r2_Score = r2_score(y_test, y_pred) # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1) # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE
    mse_score = mean_squared_error(y_test, y_pred) # MSE
    mae_score = mean_absolute_error(y_test, y_pred) # MAE
    mape_score = mean_absolute_percentage_error(y_test, y_pred) # MAPE

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

    print('\nCorrelations')
    print('Pearson: %.5f' % pearson_corr)
    print('Spearman: %.5f' % spearman_corr)
    print('Distance: %.5f' % distance_corr)


rpy2.robjects.numpy2ri.activate()
runit = robjects.r
runit['source']('/home/yhh05/smogN/smogn.R')

input_path = "/home/yhh05/smogN/Test_ebr.csv"
df = pd.read_csv(input_path, delimiter=',')
df.head()

columnsToDrop = ['Date', 'Site Id','Year','Month','Day','LE_bowen_corr(mm)',
                 'G','G-1','G-2','G-3','G-4','G-5',
                 'H','H_ebr_corr','H_ebr_corr-1','H_ebr_corr-2','H_ebr_corr-3','H_ebr_corr-4','H_ebr_corr-5',
                 'NETRAD','NETRAD-1','NETRAD-2','NETRAD-3','NETRAD-4','NETRAD-5',
                 'LE','LE_ebr_corr','ET_ebr','ET_ebr_corr','ET_ebr_corr(mm)','ETrF',
                 'Elevation','Longitude','Latitude','Climate','Vegetation',
                 'H_bowen_corr','H_bowen_corr-1','H_bowen_corr-2','H_bowen_corr-3',
                 'H_bowen_corr-4','H_bowen_corr-5', 'LE_bowen_corr','ET_bowen',
                 'ET_bowen_corr','ET_bowen_corr(mm)']

df = df.drop(columnsToDrop, axis = 1)
df.dropna(inplace=True)

df.rename(columns={"Site Id_1": "Site_1", "Site Id_2": "Site_2",
                  "Site Id_3": "Site_3", "Site Id_4": "Site_4",
                  "Site Id_5": "Site_5", "Site Id_6": "Site_6"}, inplace=True)

output_column = "LE_ebr_corr(mm)"
X = df.drop([output_column], axis = 1)
Y = df[output_column]
print(X.shape, Y.shape, "shape of dataset")
Y = np.array(Y).reshape(-1, 1)

#type desired col names
all_columns = list(X.columns)
#optional: type desired split size
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_splitting(X, Y, test_size=test_size)
X_train, X_valid, y_train, y_valid = train_test_splitting(X_train, y_train, test_size=test_size)
#standardize dataset
X_train_scaled = apply_scaling(X_train,all_columns,X_train)
X_test_scaled = apply_scaling(X_test,all_columns,X_train)
X_valid_scaled = apply_scaling(X_valid,all_columns,X_train)

utility_based = False
y_pred = fit_predict(X_train_scaled,X_test_scaled,y_train)
df_scaled_test = X_test_scaled

#combine y_test and y_pred in 1 dataset
df_scaled_test['LE_ebr_corr(mm)'] = y_test
df_scaled_test['LE_ebr_corr(mm)_pred'] = y_pred

if utility_based == True:
    error_metrics(y_test,y_pred)
else:
    evaluate(df_scaled_test, actual='LE_ebr_corr(mm)', predicted='LE_ebr_corr(mm)_pred',
             thresh=0.8, rel_method='extremes', extr_type='high',
             coef=1.5, relevance_pts=None)

