import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.optimize

import fairlearn
from fairlearn.metrics import MetricFrame
from fairlearn.datasets import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import lightgbm as lgb

# create auxilary function
def zeros_ones_to_classes(x,length = 3):
    n = int(len(x)/length)
    l = []
    for i in range(n):
        z = x[i*length:i*length+length]
        l.append(z.argmax())
    return np.array(l, dtype=int)
d = pd.read_csv('xxx.csv')
y = d.drop(['age'],axis=1)
x = d['age']
y_train,y_test,x_train,x_test = train_test_split(y,x)
lg = lgb.LGBMClassifier()
lg.fit(y_train,x_train)
lg_pred= lg.predict(y_test)

    
def write(rz):
    #output 
    fp = open('fair_algorithm_results.txt', 'w')
    fp.write('Initial Algoritmh')
    fp.write('\n\nAccuracy of the baseline predictor: '+rz['base_acc'])
    fp.write('\nAccuracy of the initail classifier: '+rz['cl_acc'])
    fp.write('\nAccuracy for male group: ' + rz['male_acc'])
    fp.write('\nAccuracy for female group: ' + rz['fem_acc'])
    fp.write('\nUnfairness of the initial algorithm:'+rz['mod_unfair'])
    fp.write('\n \n \nAfter Postprocessing with Fairness')
    fp.write('\nAccuracy for male group: '+rz['fair_male_acc'])
    fp.write('\nAccuracy for female group: '+rz['fair_fem_acc'])
    fp.write('\nUnfairness of the post-processed algorithm: '+rz['fair_mod_unfair'])
    fp.write('\n\nCompare Results:')
    fp.write('\nFairness improvement: '+rz['fair_impr'])
    fp.write('\nAccuracy_male_loss: '+rz['acc_male_loss'])
    fp.write('\nAccuracy_female_loss: '+rz['acc_fem_loss'])
    fp.close()

def run():
    # here we create male/female train/test features/labels
    female_train_features = y_train[y_train['sex_Male']==0]
    male_train_features = y_train[y_train['sex_Male']==1]
    female_train_labels = x_train[female_train_features.index]
    male_train_labels = x_train[male_train_features.index]

    female_test_features = y_test[y_test['sex_Male']==0]
    male_test_features = y_test[y_test['sex_Male']==1]
    female_test_labels = x_test[female_test_features.index]
    male_test_labels = x_test[male_test_features.index]

    # f_total,m_total stand for amount of females and males in the training set
    f_total = female_train_features.shape[0]
    m_total = male_train_features.shape[0]

    # here we create vectors of male/female probabilities. 
    #Later on they will be passed as an input to the linear programming problem
    male_train_probs = pd.DataFrame(lg.predict_proba(male_train_features)).rename(
        columns = {0:'zero_class',1:'first_class', 2:'second_class'})
    female_train_probs = pd.DataFrame(lg.predict_proba(female_train_features)).rename(
        columns = {0:'zero_class',1:'first_class', 2:'second_class'})

    m_ratio = m_total/(m_total+f_total)
    f_ratio = f_total/(m_total+f_total)
    group = int(np.sqrt(m_total+f_total))
    m_group = int(m_ratio*group)
    f_group = int(f_ratio*group)

    # here we create male and female predictor arrays
    male_predictor_array = []
    female_predictor_array = []

    # here we solve linear programming problems and create a set of male/female fair classifiers

    # create parameters for linear progrmamms; their are the same for all samples
    m = m_group
    f = f_group

    bounds = []
    for i in range(3*m+3*f):
        bounds.append((0,1))

    equation_vector = [1]*(m+f)
    for i in range(3):
        equation_vector.append(0)

    equation_matrix = np.zeros((m+f+3,3*f+3*m))
    for i in range(f+m):
        equation_matrix[i,3*i] = 1
        equation_matrix[i,3*i+1] = 1
        equation_matrix[i,3*i+2] = 1
    for i in range(3):
        for j in range(m):
            equation_matrix[f+m+i,3*j+i] = f
        for j in range(f):
            equation_matrix[f+m+i, 3*m+3*j+i]=-m
    x= 50*group

    #solving linear programm; each solution will result in one male and one female random forest
    for k in range(x):
        male_sample = np.array(male_train_probs.sample(m_group))
        female_sample = np.array(female_train_probs.sample(f_group))
        C = male_sample.ravel()
        B = female_sample.ravel()
        objective = (-1)*np.concatenate((C,B))

        array = scipy.optimize.linprog(
            c = objective, A_ub=None, b_ub=None, 
                           A_eq=equation_matrix, 
                           b_eq=equation_vector, 
            bounds=bounds, method='highs-ipm', callback=None, options=None, x0=None).x

    # finally create vectors of fair predictions
        fair_pred = zeros_ones_to_classes(array)
        fair_pred_male = fair_pred[:m]
        fair_pred_female = fair_pred[m:]

    # here we prepare classes to relabeling
        mdf = pd.DataFrame(male_sample, columns = ['zero_class', 'first_class','second_class'])
        male_features_after_classif = mdf.copy()
        mdf['fair'] = fair_pred[:m]
        fdf = pd.DataFrame(female_sample, columns = ['zero_class', 'first_class','second_class'])
        female_features_after_classif = fdf.copy()
        fdf['fair'] = fair_pred[m:]

    # create male and female random forest classifiers 
        m_predictor = DecisionTreeClassifier()
        m_predictor.fit(male_features_after_classif,mdf['fair'])
        f_predictor = DecisionTreeClassifier()
        f_predictor.fit(female_features_after_classif,fdf['fair']);
        male_predictor_array.append(m_predictor)
        female_predictor_array.append(f_predictor)

    # consider male and female test parts
    female_test_features = y_test[y_test['sex_Male']==0]
    male_test_features = y_test[y_test['sex_Male']==1]
    female_test_labels = x_test[female_test_features.index]
    male_test_labels = x_test[male_test_features.index]

    # get predictions/probabilities on male and female parts
    val_male_predictions = lg.predict(male_test_features)
    val_female_predictions = lg.predict(female_test_features)
    val_male_probs = lg.predict_proba(male_test_features)
    val_female_probs = lg.predict_proba(female_test_features)

    # prepare to create the matrices of male/female predictions
    val_male_index = male_test_features.index
    val_female_index = female_test_features.index

    male_rows = val_male_index.shape[0]
    male_cols = len(male_predictor_array)
    female_rows = val_female_index.shape[0]
    female_cols = len(female_predictor_array)

    # male matrices of predictions; each column is a result of applying one of male random forests on whole male-test set 
    male_final_array = np.empty(shape = (male_cols,male_rows))
    for i in range(male_cols):
        male_final_array[i] = male_predictor_array[i].predict(val_male_probs)
    male_final_array = pd.DataFrame(male_final_array)

    female_final_array = np.empty(shape = (female_cols,female_rows))
    for i in range(female_cols):
        female_final_array[i] = female_predictor_array[i].predict(val_female_probs)
    female_final_array = pd.DataFrame(female_final_array)

    male_final_ans = []
    for i in range(male_rows):
        male_final_ans.append(male_final_array[i].value_counts().sort_values(ascending = False).index[0])
    female_final_ans = []
    for i in range(female_rows):
        female_final_ans.append(female_final_array[i].value_counts().sort_values(ascending = False).index[0])

    #prepare variables for the output
    baseline_accuracy = d.age.value_counts().max()/d.shape[0]
    classifier_accuracy = accuracy_score(lg_pred, x_test)
    male_accuracy = accuracy_score(val_male_predictions, male_test_labels)
    female_accuracy = accuracy_score(val_female_predictions, female_test_labels)
    model_unfairness = abs(male_accuracy - female_accuracy)
    fair_male_accuracy = accuracy_score(male_final_ans,male_test_labels)
    fair_female_accuracy = accuracy_score(female_final_ans,female_test_labels)
    fair_model_unfairness= abs(fair_male_accuracy-fair_female_accuracy)
    
    results = {
        'base_acc': str(baseline_accuracy),
        'cl_acc': str(classifier_accuracy),
        'male_acc': str(male_accuracy),
        'fem_acc': str(female_accuracy),
        'mod_unfair': str(model_unfairness),
        'fair_male_acc': str(fair_male_accuracy),
        'fair_fem_acc': str(fair_female_accuracy),
        'fair_mod_unfair': str(fair_model_unfairness),
        'fair_impr': str(-fair_model_unfairness+model_unfairness),
        'acc_male_loss': str(male_accuracy-fair_male_accuracy),
        'acc_fem_loss': str(female_accuracy-fair_female_accuracy)
    }
    write(results)
