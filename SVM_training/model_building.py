# Build a classifer using positive and negative datasets.
#
# Based on the calculated features, generate training and test set.
#
#
import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, chi2, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve, auc
#from sklearn.model_selection import cross_validation
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
import joblib
#from sklearn.externals import joblib


def bacc_score(y_true, y_pred):
    all_classes = list(set(np.append(y_true, y_pred)))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_sensitivity = float(sum((y_pred == this_class) & (
            y_true == this_class))) / float(sum((y_true == this_class)))
        this_class_specificity = float(sum((y_pred != this_class) & (
            y_true != this_class))) / float(sum((y_true != this_class)))
        this_class_accuracy = (this_class_sensitivity +
                               this_class_specificity) / 2
        all_class_accuracies.append(this_class_accuracy)
    return sum(all_class_accuracies)/float(len(all_class_accuracies))


def main(arg=sys.argv):
    if len(arg) != 3:
        print(sys.stderr, "*.py all_tissues_mapped_dataset.csv precursor_only_features.txt")
        sys.exit(1)
    else:
        featureListSelected = []
        with open(arg[2], 'r') as inf:
            line = inf.readline()
            line = inf.readline()
            while line != '':
                print(line)
                featureListSelected.append((float(line.strip().split('\t')[1]), int(
                    line.strip().split('\t')[2]), line.strip().split('\t')[0]))
                line = inf.readline()
        time1 = time.time()
        print(featureListSelected)
        data = pd.read_csv(arg[1])
        #data_x_raw = data.iloc[:,1:].values
        #data_y_raw = data.iloc[:,0].values
        #featureList_raw = data.columns.values.tolist()
        #featureList_raw.insert(0, 'probability')
        #featureList_raw.insert(0, 'predictedValue')
        # Encode categorical features using one-hot-coding after deleting the 2nd and 3rd column in data
        data.drop(data.columns[[1, 2]], axis=1, inplace=True)
        data = pd.get_dummies(data)
        #featureList = data.columns.values.tolist()[1:]

        # subselect data with the selected features
        totalfeatureList = data.columns.values.tolist()
        subIndexList = [totalfeatureList.index(item[2]) for item in featureListSelected]
        # Creat a variable for the feature data
        data_x = data.iloc[:, subIndexList].values
        # Create a variable for the target data
        data_y = data.iloc[:, 0].values
        X_train = data_x
        y_train = data_y 
#        X_train, X_test, y_train, y_test = train_test_split(
#          data_x, data_y, test_size=0.20, random_state=101)
#        print([item[2] for item in featureListSelected])
#        np.save('X_train', X_train)
#        np.save('y_train', y_train)
#        np.save('X_test', X_test)
#        np.save('y_test', y_test)
        # Standardize Feature Data by removing the mean and scaling to unit variance
        sc = StandardScaler()
        sc.fit(X_train)
        x_train_std = sc.transform(X_train)
        # Remove features with low variance
        #selector = VarianceThreshold(threshold=0.0001)
        # selector.fit(x_train_std)
        #x_train_std_filter = selector.transform(x_train_std)
        x_train_std_filter = x_train_std
        # print len(featureListSelected)
        # print featureListSelected
        print('FeaturesCount\tTraining matthews corrcoef\tTime')
        bacc_scorer = make_scorer(bacc_score, greater_is_better=True)
        mcc_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
        # start to train the model use the selected features.
        pipe_svc = Pipeline(
            [('clf', SVC(probability=True, class_weight='balanced', random_state=1))])
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000]
        param_grid = [{'clf__C': param_range,
                       'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
        gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
                          scoring=mcc_scorer, cv=10, n_jobs=32)
        gs = gs.fit(x_train_std_filter, y_train)
        clf = gs.best_estimator_
        clf.fit(x_train_std_filter, y_train)
        time2 = time.time()
        print('%d\t%.3f\t%.1f' %
              (len(featureListSelected), gs.best_score_, time2-time1))
        """
		for index, item in enumerate(featureListSelected):
			if index == len(featureListSelected)-1:
				selectLabelList = [subitem[1] for subitem in featureListSelected[:index+1]]
				selectFeatureNameList = [subitem[2] for subitem in featureListSelected[:index+1]]
				x_train_std_filter_select = x_train_std_filter[:,selectLabelList]
				
				pipe_svc = Pipeline([('clf', SVC(probability=True, class_weight='balanced', random_state=1))])
				param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000]
				param_grid = [{'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
				gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=mcc_scorer, cv=10, n_jobs=6)
				gs = gs.fit(x_train_std_filter_select, data_y)
				clf = gs.best_estimator_
				clf.fit(x_train_std_filter_select, data_y)
				time2 = time.time()
				print '%d\t%.3f\t%.1f'%(index+1, gs.best_score_, time2-time1)
		"""
        joblib.dump([sc, clf, featureListSelected],
                    "mirgenedb_all_negative_annotations.pkl", compress=1)


if __name__ == "__main__":
    main()
