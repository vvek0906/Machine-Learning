#%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 12)

import numpy       as np
import pickle
import sys

from matplotlib     import pyplot as plt
from operator       import itemgetter

sys.path.append("ud120-projects-master/")

from feature_format import featureFormat, targetFeatureSplit

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
        'salary',
        'deferral_payments',
        'total_payments',
        'bonus',
        'total_stock_value',
        'loan_advances',
        ] # You will need to use more features

### Load the dictionary containing the dataset
with open("ud120-projects-master/final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# The data contains a TOTAL sample, which will confuse our classifier if we don't eliminate it.
data_dict.pop('TOTAL', 0)
#print(data_dict)    

import pandas as pd
from matplotlib.colors import ListedColormap

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# Note: It appears that pandas.scatter_matrix doesn't quite work
#       as advertised, in the documentation. If it did, this wouldn't
#       be necessary. You could pass a colormap, instead.
palette = {0 : 'blue', 1 : 'red'}
labels_c = map(lambda x: palette[int(x)], labels)

data_frame = pd.DataFrame(features, columns=features_list[1:])
grr = pd.plotting.scatter_matrix(data_frame, alpha=0.8, c=labels_c)

from operator   import itemgetter
   
for feature in ['loan_advances', 'total_payments']:
    features_list.remove(feature)

constituent_features_list = ['poi',
        'shared_receipt_with_poi',
        'expenses',
        'from_this_person_to_poi',
        'from_poi_to_this_person',
        ]

new_data = featureFormat(data_dict, constituent_features_list)
new_labels, new_features = targetFeatureSplit(new_data)
new_labels_c = map(lambda x: palette[int(x)], new_labels)

new_data_frame = pd.DataFrame(new_features, columns=constituent_features_list[1:])
grr = pd.scatter_matrix(new_data_frame, alpha=0.8, c=new_labels_c)

for key in data_dict.keys():
    features_dict = data_dict[key]
    res = 1
    for subkey in constituent_features_list[1:]:
        x = features_dict[subkey]
        if(np.isnan(float(x))):
            res = 0
        else:
            res *= x
    data_dict[key]['expenses_and_poi_contact'] = res

features_list.append('expenses_and_poi_contact')

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

labels_c = map(lambda x: palette[int(x)], labels)

data_frame = pd.DataFrame(features, columns=features_list[1:])
grr = pd.scatter_matrix(data_frame, alpha=0.8, c=labels_c)

#for feature in ['expenses_and_poi_contact']:
#    l = [(item[0], item[1][feature]) for item in data_dict.items() if not np.isnan(float(item[1][feature]))]
#    l.sort(key=itemgetter(1), reverse=True)
#    print "Top 3 values for feature, '{}': {}".format(feature, l[:3])


from sklearn                  import svm, tree
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.metrics          import precision_score, recall_score

sys.path.append("ud120-projects-master/final_project/")
from tester                   import test_classifier

print "Trying the SVM classifier..."
clf = svm.SVC()
test_classifier(clf, data_dict, features_list)

print "\nTrying the Decision Tree classifier..."
clf = tree.DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list)

print "Trying the AdaBoost classifier..."
clf = AdaBoostClassifier()
test_classifier(clf, data_dict, features_list)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import classification_report

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, stratify=labels)

# Set the parameters by cross-validation
tuned_parameters = [
    {'criterion' : ['gini', 'entropy'],
     'splitter'  : ['best', 'random'],
    },
]

scores = ['precision', 'recall']

for score in scores:
    print("Tuning hyper-parameters for %s:" % score)

    clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters,
                       scoring='%s_macro' % score)
    clf.fit(features_train, labels_train)

    print("\tBest parameters set found on development set:"),
    print(clf.best_params_)
    print("\tGrid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("\t\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("\tDetailed classification report:")
    y_true, y_pred = labels_test, clf.predict(features_test)
    print(classification_report(y_true, y_pred))

features_list.remove('deferral_payments')
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

clf = tree.DecisionTreeClassifier(splitter='best', criterion='entropy')
test_classifier(clf, data_dict, features_list)

# zip(*(...)) = transpose(...)
ys, xs = zip(*[(item[1]['poi'], item[1]['expenses_and_poi_contact']) for item in data_dict.items()])
plt.figure(figsize=(4, 4))
plt.scatter(xs, ys)

from matplotlib.colors import ListedColormap

cmap = ListedColormap(['b', 'r'])
xs, ys, zs = zip(*[(item[1]['total_stock_value'], item[1]['expenses_and_poi_contact'], item[1]['poi'])
                   for item in data_dict.items()])
plt.figure(figsize=(4, 4))
plt.scatter(xs, ys, c=zs, cmap=cmap)
plt.xlabel("Total Stock Value")
plt.ylabel("Expenses & POI Contact")
plt.title("Red dots are POIs.")
