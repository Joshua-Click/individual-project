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
    '''Reads the csv once located in local folder and turns it into a dataframe'''

    df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
    df.columns = df.columns.str.lower()
    
    return df

def acquire_diabetes_df2():
    '''Reads the csv once saved from kaggle in local folder and turns into a dataframe'''
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    df.columns = df.columns.str.lower()
    
    return df

def acquire_prep_diabetes_df3(df):
    '''Creates a balanced sample dataframe from the acqure_diabetes_df1 function'''
    prediabetic = df[df.diabetes_012 == 1]
    not_diabetic = df[df.diabetes_012 == 0].sample(4631)
    diabetic = df[df.diabetes_012 == 2].sample(4631)
    df = not_diabetic.append(prediabetic)
    df = df.append(diabetic)
    return df


# PREPARE

# lets make the split for train, validate, test
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
        stratify=df[target])
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1108,
        stratify=train_val[target])
    print(f'Train: {len(train)/len(df)}')
    print(f'Validate: {len(validate)/len(df)}')
    print(f'Test: {len(test)/len(df)}')
    return train, validate, test

# EXPLORE

def corr_heat(df, drops=None):
    '''Creates a heatmap off of the dataset
    
    arguments: df, 'drop items'
    
    returns: heatmap visualization'''
    sns.heatmap(df.drop(columns=drops).corr(), center=1)
    plt.title('Correlation Heatmap')

def column_split(df):
    '''Takes the qualitative and quantitative columns and splits them
    as such. Ex: cat_cols, num_cols = column_split(df)
    
    arguments: dataframe
    
    return: cat_cols, num_cols'''

    # separating our numeric and categorical columns:
    # initialize two empty lists for each type:
    cat_cols, num_cols = [], []
    # set up a for loop to build those lists out:
    # so for every column in explore_columns:
    for col in df:
        # check to see if its an object type,
        # if so toss it in categorical
        if df[col].dtype == 'O':
            cat_cols.append(col)
        # otherwise if its numeric:
        else:
            # check to see if we have more than just a few values:
            # if thats the case, toss it in categorical
            if df[col].nunique() < 10:
                cat_cols.append(col)
            # and otherwise call it continuous by elimination
            else:
                num_cols.append(col)

    return cat_cols, num_cols

def stacked_plot(col_to_stack, df):
    '''Takes the prepared columns from column_split function and plots
    stacked percentage graphs of each category.
    
    arguments: column list, dataframe
    
    return: visual barcharts'''
    
    for index, column in enumerate(col_to_stack):
        bar_by_cat = pd.crosstab(df[column], df['diabetes_012']).apply(lambda x: x/x.sum()*100, axis=1)
        bar_by_cat.plot(kind='bar', stacked=True)
        plt.ylabel('Percentage')
        plt.xlabel(column)

# STATS

def chi2_test(df, var1, var2):

    '''Runs a chi2 stats test for 2 variables
    
    arguments: df, var1, var2
    
    returns: print statements'''
    alpha = .05
    observed = pd.crosstab(df[var1], df[var2])
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

# MODELING

# Lets get our x_train, y_train, etc.
def next_split(train, validate, test, target):
    '''This function creates your modeling variables with the train, validate, test 
    sets and returns them
    
    argument: train, validate, test
    
    return: X_train, X_validate, X_test, y_train, y_validate, y_test'''

    X_train = train.drop(columns=[target])

    X_validate = validate.drop(columns=[target])

    X_test = test.drop(columns=[target])

    y_train = train[target]

    y_validate = validate[target] 

    y_test = test[target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def calculate_baseline_accuracy(y_train, y_validate):
    """
    Calculates the baseline accuracy for a classification problem.

    Parameters:
    y_train (pandas.Series): The training target variable.
    y_validate (pandas.Series): The validation target variable.

    Returns:
    None
    """
    # Calculate the baseline accuracy
    baseline_acc = y_train.mean()

    # Calculate the accuracy of the baseline prediction on the validation set
    baseline_pred = [y_train.mode()[0]] * len(y_validate)
    baseline_acc = accuracy_score(y_validate, baseline_pred)

    # Print the baseline accuracy on the validation set
    print(f"Baseline accuracy on validation set: {baseline_acc:.4f}")

def rforest(X_train, X_validate, y_train, y_validate):
    '''This function runs multiple random forest models up to 10 max depth and 10 min samples
    and provides them in a dataframe
    
    arguments: X_train, X_validate, y_train, y_validate 
    
    returns a pandas dataframe'''

    scores_all = []

    for x in range(1,11):
        
        # looping through min_samples_leaf front to back 
        # looping through max_depth back to front
        rf = RandomForestClassifier(random_state=7, min_samples_leaf=x, max_depth=11-x) # different if x = 10 vs x = 1
        #fit it
        rf.fit(X_train, y_train)
        #transform it
        train_acc = rf.score(X_train, y_train)
        
        #evaluate on my validate data
        val_acc = rf.score(X_validate, y_validate)
        diff_acc = train_acc - val_acc
        scores_all.append([x, 11-x, train_acc, val_acc, diff_acc])

    scores_df = pd.DataFrame(scores_all, columns =['min_samples_leaf','max_depth','train_acc','val_acc', 'diff_acc'])
    scores_df = scores_df.sort_values('diff_acc', ascending=True)
    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title("Top 15 important features")
    plt.show()

    return scores_df

def plotForest(scores_df):  
    '''graphs the random forest models from rforest function
    
    arguments: scores_df
    
    returns a matplotlib visual'''

    plt.figure(figsize=(12,6))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.val_acc, label='validate', marker='o')
    plt.xlabel('max depth and min leaf sample')
    plt.ylabel('accuracy')

    plt.xticks([1,2,3,4,5,6,7,8,9,10],
            [('1 and 10'),('2 and 9'),('3 and 8'),('4 and 7'),('5 and 6'),
            ('6 and 5'),('7 and 4'), ('8 and 3'), ('9 and 2'), ('10 and 1') ]
            )

    plt.title('Random Forest\nThe accuracy change with hyper parameter tuning on train and validate')
    plt.legend()
    plt.show()

def get_knn(X_train, X_validate, y_train, y_validate):
    '''graphs the knn models
    
    arguments: X_train, X_validate, y_train, y_validate
    
    return: a matplotlib visual'''

    k_range = range(1, 20)
    train_scores = []
    validate_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k, weights='uniform')
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        validate_scores.append(knn.score(X_validate, y_validate))
    plt.figure(figsize=(12,6))
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('KNN\nThe accuracy change with hyper parameter tuning on train and validate')
    plt.plot(k_range, train_scores, label='Train')
    plt.plot(k_range, validate_scores, label='Validate')
    plt.legend()
    plt.xticks([0,5,10,15,20])
    plt.show()

def get_knn_k(X_train, X_validate, y_train, y_validate, k):
    '''runs the knn with 15 neighbors
    
    arguments: X_train, X_validate, y_train, y_validate, k=n_neighbors
    
    return: train and validate accuracy scores'''
    k=k
    knn_k =  KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn_k.fit(X_train, y_train)
    print(f' Accuracy of KNN on train data is {knn_k.score(X_train, y_train)}')
    print(f' Accuracy of KNN on validate data is {knn_k.score(X_validate, y_validate)}')


def get_logreg(X_train, X_validate, y_train, y_validate):
    '''runs the logistic regression model
    
    arguments: X_train, X_validate, y_train, y_validate
    return: train and validate accuracy scores'''
    logit = LogisticRegression(random_state=7)
    logit.fit(X_train, y_train)

    
    print(f' Accuracy of Logistic Regression on train is {logit.score(X_train, y_train)}')
    print(f' Accuracy of Logistic Regression on validate is {logit.score(X_validate, y_validate)}')

def grid_reg(X_train, y_train):

    '''Used to find the best hyperparameters for the logistic regression models to use'''
    #Define the hyperparameters to tune
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'max_iter': [100, 500, 1000]
    }

    #Create a logistic regression model
    logreg = LogisticRegression(random_state=7)

    #Create a grid search object
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')

    #Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    #Print the best hyperparameters and the corresponding accuracy score
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy Score on Train:", grid_search.best_score_)

#-----------
def logreg_grid(X_train, X_validate, y_train, y_validate):
    '''This is specific to the logistic regression model
    that was determined by the grid_reg func for diabetes dataframes'''
    #Train a Logistic Regression Model
    logreg = LogisticRegression(max_iter=100, random_state=7, C=1, penalty='l2', solver='liblinear')
    logreg.fit(X_train, y_train)

    #Evaluate the Model on Train
    y_train_pred = logreg.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Train Set Accuracy: {train_accuracy}')

    #Evaluate the Model on Validate
    y_val_pred = logreg.predict(X_validate)
    val_accuracy = accuracy_score(y_validate, y_val_pred)
    print(f'Validate Set Accuracy: {val_accuracy}')


def get_logreg_test(X_train, X_test, y_train, y_test):
    '''get logistic regression accuracy on test data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(max_iter=100, random_state=7, C=1, penalty='l2', solver='liblinear')
    logit.fit(X_train, y_train)

    # print result
    print(f"Accuracy of Logistic Regression on test is {logit.score(X_test, y_test)}")


def select_kbest(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    
    return: a list of the selected features from the SelectKBest process
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    mask = kbest.get_support()
    return X.columns[mask]