#-------------------------------------------------------
#
# Script to analyse sales figures for Totus
# Hayden Eastwood - 30-10-2018
# Last updated: 30-10-2018
# Version: 1.0
#
#
# -------------------------------------------------------

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import json
import numpy as np
import pandas as pd
import calendar
import seaborn as sb
import time
from datetime import datetime, timedelta, date
from operator import itemgetter
import seaborn as sns
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# my module with classes
import mlhayden
print " "
print " * * * * * * * * * * * * * * * * *"
print " - Totus data challenge.         -"
print " - Hayden Eastwood               - "
print " - hayden.eastwood@gmail.com.    - "
print " * * * * * * * * * * * * * * * * *"
print " "


# ---------- Parameters ----------
print " -- Loading parameters"
dict_classifiers = {
    "Logreg": LogisticRegression(solver='lbfgs'),
    "NN": KNeighborsClassifier(),
    "LinearSVM": SVC(probability=True, kernel='linear'), #class_weight='balanced'
    "GBC": GradientBoostingClassifier(),
    "DT": tree.DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB(),
}

parameters = {
	'GBC':{
		'learning_rate': list(np.linspace(0.01, 10, 5)),
		'n_estimators': [10, 20, 30, 40, 50],
		'max_depth': [2,3,4,5],
		'min_samples_split': [2,3,4],
	},
	'RF':{
	  	'n_estimators': [2, 4, 6], 
	  	'criterion': ['entropy', 'gini'],
	  	'max_depth': [1, 2, 3, 5, 10], 
	  	'min_samples_split': [3, 6, 7],
	  	'min_samples_leaf': [1, 3, 5]
	},
	 'Logreg':{
	  	'C': [0.001,0.01,0.05, 0.1,1,10,100], 
	  	'fit_intercept': [True, False]
	},
	'DT':{
	  	'max_depth': list(range(1, 40, 2)),
	  	'min_samples_split': [2, 3, 5, 6],
	},
	'NN':{ 
		"n_neighbors": list(np.arange(1, 7))
	 },
	'LinearSVM': {
	 	'C': [0.001, 0.01, 0.1, 1, 10],
	 	'gamma': [0.001, 0.01, 0.1, 1]
	},
	'NB':{} #no hypers to optimise for NB
}
risk_quantile = 0.90

# /---------- Parameters ----------

print " -- Loading data"
with open('challenge.json', 'r') as f:
	data = pd.read_json(f)
data['is_churn']=data['is_churn'].fillna(1).astype(int)	# there are some "na" values in the churn column, which I'm assuming are 0
data2 = data
data2['is_churn']=data['is_churn'].fillna(0).astype(int)	
data3 = data.dropna()

print " -- Configuring and cleaning data"
churn = data # NB have used data with missing churns set to 1 (though I've tried the other combinations also)
churn.register_date = pd.to_datetime(churn.register_date, format='%Y-%m-%d')
churn.set_index('register_date', inplace=True)
churn = churn.sort_index()
churn['COUNTER'] = 1

print ' -- Generating "at risk" information'
print '    -- Products'
items_group = (churn[(churn.is_churn < 3)].groupby(['item_code']).agg(np.sum).reset_index())
items_group['items_to_churn_ratio'] = items_group['is_churn'] / items_group['quantity'] 
at_risk_items = items_group[(items_group.items_to_churn_ratio > items_group['items_to_churn_ratio'].quantile(risk_quantile))]['item_code'].reset_index()

print '    -- Sales channels'
sales_group = (churn[(churn.is_churn < 3)].groupby(['sales_channel']).agg(np.sum).reset_index())
sales_group['sales_to_churn_ratio'] = items_group['sales_channel'] / items_group['quantity'] 
at_risk_sales = sales_group[(sales_group.sales_to_churn_ratio > sales_group['sales_to_churn_ratio'].quantile(risk_quantile))]['sales_channel'].reset_index()

print '    -- Sellers'
sellers = (churn[(churn.is_churn < 3)].groupby(['seller_code']).agg(np.sum).reset_index())
seller_group = churn[(churn.is_churn < 4)].groupby(['seller_code','customer_code']).count().reset_index()
seller_customers = pd.value_counts(seller_group['seller_code']).reset_index()
seller_customers = seller_customers.sort_values(by=['index']).reset_index()
sellers['customer_count'] = seller_customers['seller_code']
sellers['customer_to_churn_ratio'] = sellers['is_churn'] / sellers['customer_count'] 
at_risk_sellers = sellers[(sellers.customer_to_churn_ratio > sellers['customer_to_churn_ratio'].quantile(risk_quantile))]['seller_code'].reset_index()

t_start = time.clock()

print ' -- Generating customer aggregation matrix'
print '    -- Initialising columns'
customers = (churn[(churn.is_churn < 4)].groupby(['customer_code']).agg(np.sum)).reset_index()[['customer_code']]
customers['risky_seller_transactions'] = 0
customers['risky_item_transactions'] = 0
customers['risky_sales_transactions'] = 0
customers['is_churn'] = 0

timeArray = {'transactions_last_week': 7, 'transactions_last_30': 30, 'transactions_last_90': 90, 'transactions_last_6_months': 183, 'transactions_last_year': 365}
for key, value in timeArray.iteritems():
	customers[key] = 0

print '    -- Generating aggregates'
for count in range(0, len(customers)):
	transactions = churn[(churn.customer_code == customers['customer_code'][count]) & (churn.seller_code.isin(at_risk_sellers['seller_code']))] #.groupby('seller_code').count()
	risky_item_transactions = sum(churn[(churn.customer_code == customers['customer_code'][count]) & (churn.item_code.isin(at_risk_items['item_code']))]['quantity']) #.groupby('seller_code').count()
	riskySalesChannels = sum(churn[(churn.customer_code == customers['customer_code'][count]) & (churn.sales_channel.isin(at_risk_sales['sales_channel']))]['quantity']) #.groupby('seller_code').count()
	if not transactions.empty:
		customers['risky_seller_transactions'][count] = len(transactions)
		customers['is_churn'][count] = int(sum(transactions['is_churn']))/len(transactions)
	recentDate = churn[(churn.customer_code == customers['customer_code'][count])].sort_index(ascending=False).index[0].strftime('%Y-%m-%d')
	for key, value in timeArray.iteritems():	
		dateTo = (datetime.strptime(recentDate, '%Y-%m-%d') - timedelta(days=value)).strftime('%Y-%m-%d')
		try:
			numberTransactions = len(churn[(churn.customer_code == customers['customer_code'][count]) & (churn.index <= recentDate) & (churn.index >= dateTo)])
			customers[key][count] = numberTransactions
		except:
			customers[key][count] = 0
	customers['risky_item_transactions'][count] = risky_item_transactions 
t_end = time.clock()	
t_diff = t_end - t_start

print ' -- Do grid search on classifiers to get best one'
print '    -- Set up training and test sets'
customer_identities = customers['customer_code']
sc = StandardScaler()
train, test, train_target, test_target = train_test_split(sc.fit_transform(customers.drop(['customer_code', 'is_churn'], 1)), customers.is_churn, test_size=0.3, random_state=42)

print '    -- Balancing data with SMOTE algorithm'
smote = SMOTE(ratio='minority')
train_smoted, target_smoted = smote.fit_sample(train, train_target)
recalls = {}
full_results = {}
grid_objects = {}

n_components = 6
print '    -- Performing Principal Components Analysis to get ' + str(n_components) + ' components'
pca = decomposition.PCA(n_components=n_components)
pca.fit(train_smoted)
train_smoted_pca = pca.transform(train_smoted)
test_pca = pca.transform(test)

print '    -- Performing grid search'
for model, parameter in parameters.iteritems():
	print '         -- Model: ' + str(model)
	grid_objects[model] = GridSearchCV(dict_classifiers[model], parameters[model], cv=5, scoring='recall')
	grid_objects[model] = grid_objects[model].fit(train_smoted_pca, target_smoted.tolist())
	predictions = grid_objects[model].best_estimator_.predict(test_pca)
	recalls[model] = mlhayden.perf_measure(test_target.tolist(), predictions, type='recall')
	full_results[model] = mlhayden.perf_measure(test_target.tolist(), predictions, type='full')
	print '             -- Recall: ' + str(recalls[model])
	print '             -- Full results: ' + str(full_results[model])
	print " "

print ' -- Best estimator:'
recalls_ordered = sorted(recalls.items(), key=itemgetter(1))
best_estimator = recalls_ordered[-1][0]
best_estimator_recall = recalls_ordered[-1][1]
print '        -- Model: ' + str(best_estimator)
print '        -- Recall score: ' + str(best_estimator_recall)
print '        -- Number of Principal Components used: ' + str(n_components)
print '        -- True positives: ' + str(full_results[best_estimator]['TP'])
print '        -- False positives: ' + str(full_results[best_estimator]['FP'])
print '        -- True negatives: ' + str(full_results[best_estimator]['TN'])
print '        -- False negatives: ' + str(full_results[best_estimator]['FN'])
print '        -- Estimator parameters: '
for key, value in grid_objects[best_estimator].get_params()['estimator'].get_params().iteritems():
	print '              -- ' + str(key) + ': ' + str(value)
mlhayden.plot_bar(recalls)

#recalls = {'GBC': 0.363636363636, 'NB': 0.5, 'SVM': 0.863636363636, 'Logreg': 0.636363636364, 'RF': 0.454545454545, 'DT': 0.227272727273}