import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression


# ACQUIRE
def acquire_diabetes_df1():
    df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
    return df

def acquire_diabetes_df2():
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    return df

def acquire_prep_diabetes_df3(df):
    prediabetic = df[df.diabetes_012 == 1]
    not_diabetic = df[df.diabetes_012 == 0].sample(4631)
    diabetic = df[df.diabetes_012 == 2].sample(4631)
    df = not_diabetic.append(prediabetic)
    df = df.append(diabetic)
    return df


# PREPARE




def split_data(df, target=None) -> tuple:
    '''
    split_data will split data into train, validate, and test sets
    
    if a discrete target is in the data set, it may be specified
    with the target kwarg (Default None)
    
    return: three pandas DataFrames
    '''
    train_val, test = train_test_split(
        df, 
        train_size=0.8, 
        random_state=1108,
        stratify=target)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1108,
        stratify=target)
    return train, validate, test