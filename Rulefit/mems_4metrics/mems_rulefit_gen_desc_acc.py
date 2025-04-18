#---------------------------------------------------------------------
# Importing Libraries
print('---------------------------------------------------------------------------------')
print('Importing Libraries')
print('---------------------------------------------------------------------------------')
print('')

import numpy
import time
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from imblearn.over_sampling import RandomOverSampler
import shap
from scipy.special import softmax
np.random.seed(0)
from sklearn.model_selection import train_test_split
import sklearn
#---------------------------------------------------------------------
# Defining metric equations

print('---------------------------------------------------------------------------------')
print('Defining Metric Equations')
print('---------------------------------------------------------------------------------')
print('')
def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")


def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc
def ACC_2 (TP, FN):
    ac = (TP/(TP+FN))
    return ac
def PRECISION(TP,FP):
    Precision = TP/(TP+FP)
    return Precision
def RECALL(TP,FN):
    Recall = TP/(TP+FN)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(TP,TN,FP,FN):
    BACC =(TP/(TP+FN)+ TN/(TN+FP))*0.5
    return BACC
def MCC(TP,TN,FP,FN):
    MCC = (TN*TP-FN*FP)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
      auc_avg += auc(fpr[i], tpr[i])
      counting = i+1
    return auc_avg/counting

def oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy and oversample
    x_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    x_np, y_np = oversample.fit_resample(x_np, y_np)

    # Convert back to pandas
    x_over = pd.DataFrame(x_np, columns=X_train.columns)
    y_over = pd.Series(y_np)
    return x_over, y_over

#---------------------------------------------------------------------


print('---------------------------------------------------------------------------------')
print('Generating Descriptive Accuracy Graph')
print('---------------------------------------------------------------------------------')
print('')

x_axis_mems = [3, 2, 1] 
# MEMS

### FILL UP NEW VALUES (ACCURACY) AND PLOT ###

#RF 
y_axis_RF = [0.70, 0.66, 0.59]
#DT
y_axis_DT = [0.55, 0.32, 0.30]
#SVM
y_axis_SVM = [0.64, 0.41, 0.39] 
#DNN
y_axis_DNN = [0.72, 0.58, 0.39]
# MLP
# y_axis_MLP = [0.64, 0.54, 0.50]

plt.clf()

# Plot the first line
plt.plot(x_axis_mems, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis_mems, y_axis_DT, label='DT', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis_mems, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fourth line
plt.plot(x_axis_mems, y_axis_DNN, label='DNN', color='orange', linestyle='--', marker='h')

# Plot the fifth line
# plt.plot(x_axis_mems, y_axis_MLP, label='MLP', color='cyan', linestyle='--', marker='+')

# Plot the sixth line
# plt.plot(x_axis_mems, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
# plt.plot(x_axis_mems, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Add labels and a legend
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.title('Descriptive Accuracy - RuleFit (MEMS)')
plt.xticks([1, 2, 3])
plt.legend()


# Show the plot
plt.show()
plt.savefig('/home/gummadi/spring25/new_methods/mems_Gen_DESC_ACC_rulefit.png')
plt.clf()