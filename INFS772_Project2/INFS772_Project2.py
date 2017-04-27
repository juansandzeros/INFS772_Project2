__author__ = 'jharrington'

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection as ms
from sklearn.linear_model import Ridge
import math
from sklearn.metrics import r2_score

def read_data():
    df = pd.read_csv('kddcup98.csv') 
    print df.head()
    return df

def variable_type(df, nominal_level = 5):
    categorical, numeric, nominal = [],[],[]
    for variable in df.columns.values:
        if np.issubdtype(np.array(df[variable]).dtype, int) or np.issubdtype(np.array(df[variable]).dtype, float):
            if len(np.unique(np.array(df[variable]))) <= nominal_level:
                nominal.append(variable)
            else:
                numeric.append(variable)
        else:
            categorical.append(variable)
    return numeric,categorical,nominal

def variable_with_missing(df):
    var_with_missing = []
    col_names = df.columns.tolist()
    for variable in col_names:
        percent = float(sum(df[variable].isnull()))/len(df.index)
        print variable+":", percent
        if percent != 0:
            var_with_missing.append(variable)
    return var_with_missing

def num_missing_mean_median(df, variable, prefix="", mean=True):
    indicator = ""
    if prefix=="":
        indicator = variable+ "_" + "missing"
    else:
        indicator = prefix + "_"+ "missing"
    df[indicator] = np.where(df[variable].isnull(),1,0)
    replaceValue = 0
    if mean== True:
        replaceValue = df[variable].mean()
    else:
        replaceValue = df[variable].median()
    df[variable].fillna(replaceValue, inplace= True)
    return df

def add_log_transform(df, variable, indexplus = 1):
    var_string = "log_" + variable
    index = df.columns.tolist().index(variable) + indexplus
    if 0.0 in df[variable].values.astype(float):
        df.insert(index, var_string, np.log10(df[variable] + 1))
    else:
        df.insert(index, var_string, np.log10(df[variable]))

def dummy_coding_for_vars(df, list_of_variables, dummy_na=False, drop_first = False, prefix=None):
    if prefix==None:
        prefix = list_of_variables
    outputdata = pd.get_dummies(df, columns=list_of_variables, prefix= prefix, dummy_na=dummy_na, drop_first=drop_first)
    return outputdata

def main():
##Task 1 Classification
    # Step 1. Import data
    # read data
    df = read_data()
    # remove duplicates
    df = df.drop_duplicates()
    # remove rows with dependent variable missing
    df = df.dropna(subset=['TARGET_B'])
    # Remove variable not needed
    df.drop("ID", axis=1, inplace=True)
    # make copy of TARGET_D to be used in Task 2
    target_d = df["TARGET_D"].copy() 
    # Drop 'TARGET_D' variable
    df.drop("TARGET_D", axis=1, inplace=True)

    # Step 2. Explore data
    # 2.1. Get variable names
    print "variable names:"
    col_names = df.columns.tolist()
    print col_names

    # 2.2. Classify variables into numeric, categorical (with strings), and nominal
    #print df.dtypes
    # the DemMedIncome and DemMedHomeValue are currency as strings, convert them to int for correct classification
    df["DemMedIncome"] = df["DemMedIncome"].replace('[\$,]','',regex=True).astype(int)
    df["DemMedHomeValue"] = df["DemMedHomeValue"].replace('[\$,]','',regex=True).astype(int)
    numeric,categorical,nominal = variable_type(df) 
    print "numeric:", numeric 
    print "categorical:", categorical 
    print "nominal:", nominal 

    # 2.3. Draw histogram for numeric variables
    df[numeric].hist()
    plt.show()

    # 2.4. quality check 
    # check for false or unreasonable values and set to missing (nan)
    # unlikely that donors are ages under 15
    df['DemAge'].replace([14,7,6,4,2,0],np.nan,inplace=True)
   
    # 2.4. Identify numeric variables that have skewed distribution and need to be log transformed 
    # check for skewness using skew and kurtosis
    # can also get an idea of skewness also by comparing describe mean with 50% value
    skew = df[numeric].skew()
    print "skewness values:"
    print skew 
    skew = df[numeric].kurt()
    print "kurtosis values:"
    print skew 

    variables_needs_tranform = ['GiftCnt36','GiftCntAll','GiftCntCard36', 'GiftCntCardAll','GiftAvgLast','GiftAvg36','GiftAvgAll','GiftAvgCard36','PromCnt12','DemMedHomeValue'] #df[skew>1 | skew<-1]
    print "transform variables:"
    print variables_needs_tranform
  
    # 2.5.
    print "2.5"
    print "categorical frequency:"
    for column in categorical:
        print df[column].value_counts(dropna=False)
    print "nominal frequency:"
    for column in nominal:
        print df[column].value_counts(dropna=False)

    # Step 3. Transform variables
    # Identify variables with missing values:
    variables_with_na = variable_with_missing(df) 
    numeric_with_na = variable_with_missing(df[numeric]) # find numeric variables with missing values and add the variables to the list numeric_with_na
    categorical_with_na = variable_with_missing(df[categorical]) # find categorical variables with missing values and add the variables to the list categorical_with_na    
    nominal_with_na = variable_with_missing(df[nominal]) # find nominal variable with missing values and add the variables to the list nominal_with_na. 
    print "variables na:"
    print variables_with_na
    print "numeric na:"
    print numeric_with_na
    print "categorical na:"
    print categorical_with_na
    print "nominal na:"
    print nominal_with_na

    # 3.1. Do categorical/nominal variable recoding and missing value imputation using dummy variables, as we discussed in the course.
    df = dummy_coding_for_vars(df,categorical)

    # 3.2. For continuous variables with skewed distributions, we do log transformation 
    for variable in variables_needs_tranform:
        df["log_"+variable] = (df[variable].fillna(0)+1).apply(np.log)

    # 3.3. Do missing value imputation for continuous variables (replacing miss values with median and add an indicator)
    for column in numeric_with_na:
        num_missing_mean_median(df, column)

    # after transformation and missing value imputation, we clean our data. 
    independent_vars = ['GiftCnt36', 'log_GiftCnt36', 'GiftCntAll', 'log_GiftCntAll', 'GiftCntCard36', 'log_GiftCntCard36', 'GiftCntCardAll', 'log_GiftCntCardAll', 'GiftAvgLast', 'log_GiftAvgLast', 'GiftAvg36', 'log_GiftAvg36', 'GiftAvgAll', 'log_GiftAvgAll', 'GiftAvgCard36', 'log_GiftAvgCard36', 'GiftTimeLast', 'GiftTimeFirst', 'PromCnt12', 'log_PromCnt12', 'PromCnt36', 'PromCntAll', 'PromCntCard12', 'PromCntCard36', 'PromCntCardAll', 'StatusCatStarAll', 'DemCluster', 'DemAge', 'DemMedHomeValue', 'log_DemMedHomeValue', 'DemPctVeterans', 'DemMedIncome', 'StatusCat96NK_A', 'StatusCat96NK_E', 'StatusCat96NK_F', 'StatusCat96NK_L', 'StatusCat96NK_N', 'StatusCat96NK_S', 'DemGender_F', 'DemGender_M', 'DemGender_U', 'DemHomeOwner_H', 'DemHomeOwner_U', 'GiftCntAll_missing', 'GiftAvgCard36_missing', 'DemAge_missing']
    dependent_var = 'TARGET_B'
    df_task2 = df.copy() # create a copy of the dataframe to be used in task 2
  
    # Step 4. Data partitioning (80/20)
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    train_X, test_X, train_y, test_y = train_test_split(df[independent_vars], df[dependent_var], test_size=0.2, random_state=123)

    # Step 5. Variable selection using RFECV 
    train_X_new = None 
    test_X_new = None

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear", cache_size=7000, max_iter=5000)
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(train_y,5), scoring='r2')
    selector = rfecv.fit(train_X, train_y)

    print selector.support_
    print selector.ranking_
    print("Optimal number of features : %d" % rfecv.n_features_)

    train_X_new = selector.transform(train_X)
    test_X_new = selector.transform(test_X) # for the test dataset, you need to also keep just the selected varibles
    print "selected variables:"
    all_features = [x for x in train_X.columns]
    selected_variables = [f for f, s in zip(all_features, selector.support_) if s]
    print selected_variables

    # Step 6. Model fitting and comparison. 

    # prepare cross validation folds
    num_folds = 5
    kfold = ms.StratifiedKFold(n_splits=num_folds)

    # fit Logistic Regression
    parameters = {'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    model = GridSearchCV(LogisticRegression(penalty='l2', class_weight="balanced"), parameters, cv=kfold, scoring= "accuracy")
    print model
    model.fit(train_X_new, train_y)
    pred_y = model.predict(test_X_new)
    print "The logistic regression classification results:"
    for line in metrics.classification_report(test_y, pred_y).split("\n"):
        print line
    print "Accuracy for logistic regression:"
    print(metrics.accuracy_score(test_y,pred_y))

    # fit SGD
    parameters = [{'alpha': np.arange(1,1111)/10000.0}]
    model = GridSearchCV(SGDClassifier(loss='hinge', penalty='l2'), parameters, cv=kfold, scoring='accuracy')
    print model
    model.fit(train_X_new, train_y)
    pred_y = model.predict(test_X_new)
    print "The SGD classification results:"
    print(metrics.classification_report(test_y, pred_y))
    print "Accuracy for SGD:"
    print(metrics.accuracy_score(test_y,pred_y))
    
    # fit Random Forest
    parameters = [{'n_estimators': np.arange(10,51)}]
    model = GridSearchCV(RandomForestClassifier(max_features='auto'), parameters, cv=kfold, scoring='accuracy')
    print model
    model.fit(train_X_new, train_y)
    pred_y = model.predict(test_X_new)
    print "The random forest classification results:"
    print(metrics.classification_report(test_y, pred_y))
    print "Accuracy for Random Forest:"
    print(metrics.accuracy_score(test_y,pred_y))

## Task 2 Regression
    # Step 1. 
    df_task2["TARGET_D"] = target_d
    # remove rows with dependent variable missing
    df_task2 = df_task2.dropna(subset=['TARGET_D'])
    # Drop 'TARGET_B' variable
    df_task2.drop("TARGET_B", axis=1, inplace=True)

    # Step 5. Variable selection using RFECV 
    train_X_new = None 
    test_X_new = None

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear", cache_size=7000, max_iter=5000)
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=svc, step=1, cv=ms.KFold(n_splits=5), scoring='r2')
    selector = rfecv.fit(train_X, train_y)

    print selector.support_
    print selector.ranking_
    print("Optimal number of features : %d" % rfecv.n_features_)

    train_X_new = selector.transform(train_X)
    test_X_new = selector.transform(test_X) # for the test dataset, you need to also keep just the selected varibles
    print "selected variables:"
    all_features = [x for x in train_X.columns]
    selected_variables = [f for f, s in zip(all_features, selector.support_) if s]
    print selected_variables

    # fit Ridge Regression model
    parameters = [{'alpha': np.arange(1,11)/10.0}]
    kfold = ms.KFold(n_splits=5)
    model = GridSearchCV(Ridge(max_iter=100,), parameters, cv=kfold, scoring='r2')
    print model
    model.fit(train_X_new, train_y)
    pred_y = model.predict(test_X_new)
    
    # The root mean square error
    mse = np.mean( (pred_y - test_y) ** 2)
    rmse = math.sqrt(mse)
    print ("RMSE: %.2f" % rmse)
    r2 = r2_score(pred_y, test_y)
    print ("R2 value: %.2f" % r2)
   

if __name__ == "__main__":
    main()
