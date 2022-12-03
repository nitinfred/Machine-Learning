# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split, cross_val_score


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.lines import Line2D

# Load Dataset 
df = pd.read_csv("data/wgm.csv")
df = df.loc[:, ['Age', 'age_var2',  'age_var3', 'Gender', 'Education', 'EMP_2010', 'Household_Income', 'wbi', 'MH1', 'MH6', 'Subjective_Income', 'MH7A']]
print(df)

print(df.dtypes)

df = df.loc[:, ['Age', 'age_var2',  'age_var3', 'Gender', 'Education', 'EMP_2010', 'Household_Income', 'wbi', 'MH1', 'MH6', 'Subjective_Income', 'MH7A']]

print(df)

# remove empty entries
df.drop(df.index[df['Household_Income'] == ' '], inplace = True)
df.drop(df.index[df['EMP_2010'] == ' '], inplace = True)
df.drop(df.index[df['MH7A'] == ' '], inplace = True)

# Transform categorical data ordinary data by mapping the values to integers
df_new = pd.DataFrame() # define a new data frame and populate it with features in integer format

# Age
df_new['Age'] = df['Age']

# Age Cohort
conditions = [df['age_var3'] == '15-24', df['age_var3']== '25-34', df['age_var3']== '35-49', df['age_var2']== '50-64', df['age_var2']== '65+', df['age_var2']== 'DK/Refused']
values = [1, 2, 3, 4, 5, 99]
age_cohort = np.select(conditions, values, 'New')
pd.Series(age_cohort)
df_new['Age_Cohort'] = age_cohort

# Gender
conditions = [df['Gender'] == 'Male', df['Gender']== 'Female']
values = [0, 1]
gender = np.select(conditions, values, 'New')
pd.Series(gender)
df_new['Gender'] = gender

# Education
conditions = [df['Education'] == 'Elementary or less', df['Education']== 'Secondary', df['Education']== 'Tertiary',df['Education']== 'DK/Refused']
values = [1, 2, 3, 99]
education = np.select(conditions, values, 'New')
pd.Series(education)
df_new['Education'] = education

# EMP_2010
conditions = [df['EMP_2010'] == 'Employed full time for an employer', df['EMP_2010']== 'Employed full time for self', df['EMP_2010']== 'Employed part time do not want full time',
df['EMP_2010']== 'Unemployed', df['EMP_2010']== 'Employed part time want full time', df['EMP_2010']== 'Out of workforce']
values = [1,2,3,4,5,6]
emp = np.select(conditions, values, 'New')
pd.Series(emp)
df_new['Employment_Status'] = emp

# Household income
conditions = [df['Household_Income'] == 'Poorest 20%', df['Household_Income']== 'Second 20%', df['Household_Income']== 'Middle 20%', df['Household_Income']== 'Fourth 20%', df['Household_Income']== 'Richest 20%']
values = [5, 4, 3, 2, 1]
h_income = np.select(conditions, values, 'New')
pd.Series(h_income)
df_new['Household_Income'] = h_income

# wbi
conditions = [df['wbi'] == 'Low income', df['wbi']== 'Lower-middle income', df['wbi']== 'Upper-middle income', df['wbi']== 'High income']
values = [4, 3, 2, 1]
wbi = np.select(conditions, values, 'New')
pd.Series(wbi)
df_new['Country_Income_Level'] = wbi

# MH1
conditions = [df['MH1'] == 'More important', df['MH1']== 'As important', df['MH1']== 'Less important', df['MH1']== 'DK/Refused']
values = [0, 0, 1, 99]
mh1 = np.select(conditions, values, 'New')
pd.Series(mh1)
df_new['Mental_Health_Importance'] = mh1

# MH6
conditions = [df['MH6'] == 'Yes', df['MH6']== 'No', df['MH6']== 'DK/Refused']
values = [1, 0, 99]
mh6 = np.select(conditions, values, 'New')
pd.Series(mh6)
df_new['MH6'] = mh6

# Subjective Income
conditions = [df['Subjective_Income'] == 'Living comfortably on present income', df['Subjective_Income']== 'Getting by on present income', df['Subjective_Income']== 'Finding it difficult on present income', 
df['Subjective_Income']== 'Finding it very difficult on present income', df['Subjective_Income']== 'DK', df['Subjective_Income']== 'Refused']
values = [1, 2, 3, 4, 99, 99]
subjective_income = np.select(conditions, values, 'New')
pd.Series(subjective_income)
df_new['Subjective_Income'] = subjective_income

# MH7A
conditions = [df['MH7A'] == 'Yes', df['MH7A']== 'No', df['MH7A']== 'DK/Refused']
values = [1, 0, 99]
mh7a = np.select(conditions, values, 'New')
pd.Series(mh7a)
df_new['Anxious_Or_Depressed'] = mh7a

df_new = df_new[[ 'Age', 'Age_Cohort', 'Gender', 'Education',  'EMP_2010', 'Household_Income', 'wbi', 'MH1', 'MH6', 'Subjective_Income',  'MH7A']]
print(df_new.head())
df_new.dtypes
df_new['Subjective_Income'] = pd.to_numeric(df_new['Subjective_Income'])

df_new['Education'] = pd.to_numeric(df_new['Education'])

# Feature Engineering-------------------------------

# convert to numeric
df_new['Age'] = pd.to_numeric(df_new['Age'])
df_new['Age_Cohort'] = pd.to_numeric(df_new['Age_Cohort'])
df_new['Gender'] = pd.to_numeric(df_new['Gender'])
df_new['Household_Income'] = pd.to_numeric(df_new['Household_Income'])
df_new['wbi'] = pd.to_numeric(df_new['wbi'])
df_new['EMP_2010'] = pd.to_numeric(df_new['EMP_2010'])
df_new['MH1'] = pd.to_numeric(df_new['MH1'])
df_new['MH6'] = pd.to_numeric(df_new['MH6'])
df_new['MH7A'] = pd.to_numeric(df_new['MH7A'])


# clear redundant data
df_new.drop(df_new.index[df_new['Age'] == 100], inplace = True)
df_new.drop(df_new.index[df_new['Age_Cohort'] == 99], inplace = True)
df_new.drop(df_new.index[df_new['Education'] == 99], inplace = True)
df_new.drop(df_new.index[df_new['MH1'] == 99], inplace = True)
df_new.drop(df_new.index[df_new['MH6'] == 99], inplace = True)
df_new.drop(df_new.index[df_new['Subjective_Income'] == 99], inplace = True)


df_new.drop(df_new.index[df_new['MH7A'] == 99], inplace = True)

# delete null entries
df_new = df_new.dropna()

df_new.dtypes

#df.columns[df.isin([25]).any()]
df_y = df_new['MH7A']
df_x = df_new.drop(columns=['MH7A'])
df_new = pd.concat([df_x, df_y], axis=1)
print(df_new)

# df_new.columns[df_new.isin([99]).any()]

df_y = df_new['MH7A']
df_X = df_new.drop(columns=['MH7A'])
df_new = pd.concat([df_X, df_y], axis=1)
df = df_new
print(df)

# Train-Test Split
X = df_X.to_numpy() 
y = df_y.to_numpy()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# Plot Coefficients 
def plot_coeff(model):
    data = model.coef_[0]
    plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    plt.barh(np.arange(data.size),data)
    ax.set_yticks(np.arange(data.size))
    ax.set_yticklabels(labels)
    plt.show()

# Accuracy Score
def get_accuracy_score(model,  ypred_train, ypred_test):
    print("Training Accuracy: " , accuracy_score(ytrain, ypred_train))
    print("Test Accuracy: " , accuracy_score(ytest, ypred_test))

# Plot Confusion Matrix
def plot_confusion_matrix(model, ytest, ypred):
    cm = confusion_matrix(ytest,ypred) #y = ytest
    color = 'white'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

# Plot ROC Curve
def plot_roc_curve(model, model_type, Xtest, ytest):
    plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

    if model_type == "ridge":
        y_scores = model._predict_proba_lr(Xtest) 
    else:
        y_scores = model.predict_proba(Xtest)
   
    fpr, tpr, threshold = roc_curve(ytest, y_scores[:, 1])

    plt.plot(fpr,tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green',linestyle='--')
    plt.show()

# Use optimal polynomial degree observed from Cross Validation
def find_penalty(data, classifier, range):
    X = data
    # Error bar for C values
    mean_error_train=[]
    std_error_train=[]
    mean_error=[]
    std_error=[]

    for c in range:
        if classifier == "logistic":
            model = LogisticRegression(penalty='l2', max_iter = 1000, C=c)
        else:
            model = RidgeClassifier(alpha=1/(2*c))
        temp_train = []
        temp=[]

        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred_train = model.predict(X[train])
            ypred = model.predict(X[test])

            temp_train.append(accuracy_score(y[train],ypred_train))
            temp.append(accuracy_score(y[test],ypred))

        # print(model.intercept_, model.coef_)
        mean_error_train.append(np.array(temp_train).mean())
        std_error_train.append(np.array(temp_train).std())
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    label_size = 20
    plt.figure(figsize=(30,10))
    plt.rc('axes', labelsize=label_size) 
    plt.errorbar(range, mean_error, yerr=std_error)
    plt.errorbar(range, mean_error_train, yerr=std_error_train)
    plt.xlabel('Penalty (C)')
    plt.ylabel('Accuracy Score')
    plt.xlim((-0.5,2))
    plt.legend(handles=[Line2D([], [], c="orange", label="Train"),Line2D([], [], c="blue", label="Test"),])
    plt.show()

# Logistic Regression Classifier
model_logistic = LogisticRegression(penalty='l2', max_iter=1000, C = 100).fit(Xtrain, ytrain)
print(model_logistic.intercept_, model_logistic.coef_)

# K-Fold Cross Validation for range of C values
find_penalty(X, classifier = "logistic", range = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 75, 100])

model_logistic_optimal = LogisticRegression(penalty='l2',max_iter=1000, C=0.01).fit(Xtrain, ytrain)

labels = list(df_X.columns.values)

# Plot Coefficients 
plot_coeff(model_logistic_optimal)

# Accuracy Score for Logistic Regression
model = model_logistic_optimal
ypred_train = model.predict(Xtrain)
ypred_test = model.predict(Xtest)
get_accuracy_score(model,  ypred_train, ypred_test)

# Confusion Matrix for Logistic Regression
plot_confusion_matrix(model, ytest, ypred_test)

# ROC Curve for Logistic Regression
plot_roc_curve(model, "logistic", Xtest, ytest)

# Performance Metrics for Logistic Regression model
print("Training Data")
print(classification_report(ytrain, ypred_train))
print("Test Data")
print(classification_report(ytest, ypred_test))

# Ridge Classifier
C = 0.1
model_ridge = RidgeClassifier(alpha=1/(2*C)).fit(Xtrain, ytrain)
print(model_ridge.intercept_, model_ridge.coef_)

# K-Fold Cross Validation for range of C values
find_penalty(X, classifier = "ridge", range = [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 50])

C = 0.05
model_ridge_optimal = RidgeClassifier(alpha=1/(2*C)).fit(Xtrain, ytrain)

# Plot Coefficients 
plot_coeff(model_ridge_optimal)

# Accuracy Score for Ridge Regression
model = model_ridge_optimal
ypred_train = model.predict(Xtrain)
ypred_test = model.predict(Xtest)
get_accuracy_score(model,  ypred_train, ypred_test)

# Confusion Matrix for Ridge Regression
plot_confusion_matrix(model, ytest, ypred_test)

# ROC Curve for Ridge Regression
plot_roc_curve(model, "ridge", Xtest, ytest)

# Performance Metrics for Ridge Regression model
print("Training Data")
print(classification_report(ytrain, ypred_train))
print("Test Data")
print(classification_report(ytest, ypred_test))

# Kernelized-SVM 
model_svc_sig = SVC(C = 0.1, kernel='sigmoid', degree=3, probability=True).fit(Xtrain, ytrain)

print(model_svc_sig.intercept_, model_svc_sig.dual_coef_)

# Accuracy Score for Kernelised-SVM
model = model_svc_sig
ypred_train = model.predict(Xtrain)
ypred_test = model.predict(Xtest)
get_accuracy_score(model,  ypred_train, ypred_test)

# Confusion Matrix for Kernelised-SVM
plot_confusion_matrix(model, ytest, ypred_test)

# ROC Curve for Kernelised-SVM
plot_roc_curve(model, "svm", Xtest, ytest)

# Performance Metrics for Kernelised-SVM
print("Training Data")
print(classification_report(ytrain, ypred_train))
print("Test Data")
print(classification_report(ytest, ypred_test))

# Baseline Classifier - Random
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

model_baseline_rand = DummyClassifier(strategy='uniform')
model_baseline_rand.fit(Xtrain, ytrain)

ypred_test = model_baseline_rand.predict(Xtest)

# Accuracy for Baseline model 
print(accuracy_score(ytest, ypred_test))

model = model_baseline_rand
# Confusion Matrix for Baseline
plot_confusion_matrix(model, ytest, ypred_test)

# ROC Curve for Baseline
plot_roc_curve(model, "baseline", Xtest, ytest)

# Performance Metrics for Baseline
print("Training Data")
print(classification_report(ytrain, ypred_train))
print("Test Data")
print(classification_report(ytest, ypred_test))

# Baseline Classifier - Most Frequent
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

model_baseline_freq = DummyClassifier(strategy='most_frequent')
model_baseline_freq.fit(Xtrain, ytrain)
ypred_test = model_baseline_freq.predict(Xtest)

# Accuracy for Baseline model 
print(accuracy_score(ytest, ypred_test))

model = model_baseline_freq
# Confusion Matrix for Baseline
plot_confusion_matrix(model, ytest, ypred_test)

# ROC Curve for Baseline
plot_roc_curve(model, "baseline", Xtest, ytest)

# Performance Metrics for Baseline
print("Training Data")
print(classification_report(ytrain, ypred_train))
print("Test Data")
print(classification_report(ytest, ypred_test))