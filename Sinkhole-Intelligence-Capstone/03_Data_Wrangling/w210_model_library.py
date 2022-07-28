from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, decomposition, datasets
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import PrecisionRecallDisplay

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math
import random

def print_confusion_matrix(Y_dev, Prediction, title):
    cfm = confusion_matrix(Y_dev,Prediction)
    start = 0 #It does not work
    size = 8
    if np.unique(Y_dev).max() > 5:
        size = 6
    else: 
        size = 3
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(cfm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[1]):
            ax.text(x=j+start, y=i+start,s=cfm[i, j], va='center', ha='center', size='xx-large')
    ax.set_title(title)
    

def modelresults(train_data, train_labels, test_data, test_labels, scaler, model):
    '''
    scaler: Standard, MinMax
    '''
    
    if scaler != 'none':
        pipeline = Pipeline(steps=[('t', scaler), ('m', model)])
    else:
        pipeline = Pipeline(steps=[('m', model)])
        
    pipeline.fit(train_data, train_labels)
    
    dfResults = pd.DataFrame()
    dfPred = {}
    
    scenarios = ["Train", "Test"]
    accuracy1 = []
    precision = []
    recall = []
    f1_score_list = []
    f1_scr_weighted = []
    
    for scenario in scenarios:
        
        data = []
        labels = []
        
        if scenario == "Train":
            data = train_data
            labels = train_labels
        else:
            data = test_data
            labels = test_labels
        
        predictions = pipeline.predict(data)
        
        accuracy1.append(accuracy_score(labels, predictions)) 
        precision.append(precision_score(labels, predictions))
        recall.append(recall_score(labels, predictions))
        f1_score_list.append(f1_score(labels, predictions))
        f1_scr_weighted.append(metrics.f1_score(labels, predictions, average="weighted"))
        
        dfPred[scenario] = predictions

    dfResults["Scenarios"] = scenarios
    dfResults["Accuracy1"] = accuracy1
    dfResults["F1_score"] = f1_score_list
    dfResults["F1_score_Weighted"] = f1_scr_weighted
    dfResults["Precision"] = precision
    dfResults["Recall"] = recall
 
    return(dfResults, dfPred, pipeline)


def modelresults_2(train_data, train_labels, test_data, test_labels, scaler, model):
    '''
    scaler: Standard, MinMax
    '''
    
    if scaler != 'none':
        pipeline = Pipeline(steps=[('t', scaler), ('m', model)])
    else:
        pipeline = Pipeline(steps=[('m', model)])
        
    pipeline.fit(train_data, train_labels)
    
    dfResults = pd.DataFrame()
    dfPred = {}
    
    scenarios = ["Train", "Test"]
    accuracy1 = []
    precision = []
    recall = []
    f1_score_list = []
    f1_scr_weighted = []
    
    for scenario in scenarios:
        
        data = []
        labels = []
        
        if scenario == "Train":
            data = train_data
            labels = train_labels
        else:
            data = test_data
            labels = test_labels
        
        predictions = pipeline.predict(data)
        
        accuracy1.append(accuracy_score(labels, predictions)) 
        precision.append(precision_score(labels, predictions))
        recall.append(recall_score(labels, predictions))
        f1_score_list.append(f1_score(labels, predictions))
        f1_scr_weighted.append(metrics.f1_score(labels, predictions, average="weighted"))
        
        dfPred[scenario] = predictions
    
    results = []
    
    index = ["Accuracy", "Precision", "Recall", "F1_score", "F1_score_weighted"]
    results = [accuracy1, precision, recall, f1_score_list, f1_scr_weighted]
    
    dfResults = pd.DataFrame(results, columns=scenarios, index=index)

    return(dfResults, dfPred, pipeline)


# Crossvalidation Search
def crossvalidation(scaler, model, parameters, train_data, train_labels, cv, scoring):
    
    pipeline = Pipeline([( "scaler" , scaler),
                         ("mod", model)])

    # initialize
    grid_pipeline = GridSearchCV(pipeline,
                                 parameters, 
                                 scoring=scoring,
                                 cv=cv)

    # fit
    grid_pipeline.fit(train_data,train_labels)

    return grid_pipeline


# Assign an Risk Label (High, Medium, or Low) based on H and M limits
def assignRisk(prb, H, M):
    if prb >= H:
        return("High")
    elif prb >= M:
        return("Medium")
    else: 
        return("Low")
    
    
# Generate a new Prediction base on a new probability level for Y=1 event (cut)
def newPred(prb, cut):
    if prb >= cut:
        return(1)
    else:
        return(0)
    
    
# Function to Create DataFrame with High, Medium, Low Risk Distribution for a given data
def riskdistribution(data, labels, groups, predictions, high, mid, pipeline):
    
    dfprob = pd.DataFrame()
    dfprob["No_SH"] = pipeline.predict_proba(data)[:,0]
    dfprob["SH"] = pipeline.predict_proba(data)[:,1]
    dfprob["Label"] = labels
    dfprob["Group"] = groups
    dfprob["Prediction"] = predictions
    dfprob["Risk"] = dfprob.apply(lambda row: assignRisk(row["SH"], high, mid), axis=1)
    dfprob["New_Pred"] = dfprob.apply(lambda row: newPred(row["SH"], mid), axis=1)
    dfprob['Value'] = 1
    
    return (dfprob)