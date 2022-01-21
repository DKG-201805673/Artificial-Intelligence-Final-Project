import pandas as pd

#replace path to prepared dataset
df = pd.read_csv([your filepath here])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X = standardizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

#lines 17-64 due to: https://www.learndatasci.com/glossary/binary-classification/
models = {}

#Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

#Support Vector Machines
from sklearn.svm import SVC
models['Support Vector Machines'] = SVC()

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

#Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

accuracy, precision, recall, confusion = {}, {}, {}, {}

for key in models.keys():
    
    #Fit the classifier model
    models[key].fit(X_train, y_train)
    
    #Prediction 
    predictions = models[key].predict(X_test)
    
    #Calculate Accuracy, Precision and Recall Metrics as well as Confusion Matrix
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    confusion[key] = confusion_matrix(predictions, y_test)

#create dataframe with default values
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model['Confusion Matrix'] = confusion.values()

optimized_models= {}
accuracy_opt, precision_opt, recall_opt, confusion_opt = {}, {}, {}, {}


#lines 71-88 due to: https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/
from sklearn.model_selection import GridSearchCV
#define evaluation
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


#tuning SVM
#define search space
params = dict()
params['C'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
params['gamma'] = ['scale', 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
params['degree'] = [1, 3, 5]
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']

#define the search
search = GridSearchCV(SVC(), params, scoring='accuracy', n_jobs=-1, cv=cv)
#perform the search
search.fit(X_train, y_train)
#print and store the best result
print(f'Optimized SVM score: {search.best_score_} with parameters: {search.best_params_}.')
optimized_models['Optimized Support Vector Machines'] = (search.best_score_, search.best_params_)
#make predictions
predictions = search.predict(X_test)
#Calculate Accuracy, Precision and Recall Metrics as well as Confusion Matrix
accuracy_opt['Optimized Support Vector Machines'] = accuracy_score(predictions, y_test)
precision_opt['Optimized Support Vector Machines'] = precision_score(predictions, y_test)
recall_opt['Optimized Support Vector Machines'] = recall_score(predictions, y_test)
confusion_opt['Optimized Support Vector Machines'] = confusion_matrix(y_test, predictions)


#tuning Logistic Regression:
#define search space
params = dict()
params['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
params['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
params['C'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

#define the search
search = GridSearchCV(LogisticRegression(), params, scoring='accuracy', n_jobs=-1, cv=cv)
#perform the search
search.fit(X_train, y_train)
#print and store the best result
print(f'Optimized Logistic Regression score: {search.best_score_} with parameters: {search.best_params_}.')
optimized_models['Optimized Logistic Regression'] = (search.best_score_, search.best_params_)
#make predictions
predictions = search.predict(X_test)
#Calculate Accuracy, Precision and Recall Metrics as well as Confusion Matrix
accuracy_opt['Optimized Logistic Regression'] = accuracy_score(predictions, y_test)
precision_opt['Optimized Logistic Regression'] = precision_score(predictions, y_test)
recall_opt['Optimized Logistic Regression'] = recall_score(predictions, y_test)
confusion_opt['Optimized Logistic Regression'] = confusion_matrix(y_test, predictions)


#tuning Decision Trees
#define search space
params = dict()
params['criterion'] = ['gini', 'entropy']
params['splitter'] = ['random', 'best']
params['max_depth'] = [None,2,4,6,8,10,12,14,16]
#define the search
search = GridSearchCV(DecisionTreeClassifier(), params, scoring='accuracy', n_jobs=-1, cv=cv)
#perform the search
search.fit(X_train, y_train)
#print and store the best result
print(f'Optimized Decision Tree score: {search.best_score_} with parameters: {search.best_params_}.')
optimized_models['Optimized Decision Trees'] = (search.best_score_, search.best_params_)
#make predictions
predictions = search.predict(X_test)
#Calculate Accuracy, Precision and Recall Metrics as well as Confusion Matrix
accuracy_opt['Optimized Decision Trees'] = accuracy_score(predictions, y_test)
precision_opt['Optimized Decision Trees'] = precision_score(predictions, y_test)
recall_opt['Optimized Decision Trees'] = recall_score(predictions, y_test)
confusion_opt['Optimized Decision Trees'] = confusion_matrix(y_test, predictions)


#tuning Random Forest
#define search space
params = dict()
params['min_samples_split'] = [2, 3, 5, 10]
params['n_estimators'] = [100, 300]
params['max_depth'] = [None, 3, 5, 15, 25]
params['max_features'] = ['auto', 3, 5, 10, 20]
params['criterion'] = ['gini', 'entropy']
#define the search
search = GridSearchCV(RandomForestClassifier(), params, scoring='accuracy', n_jobs=-1, cv=cv)
#perform the search
search.fit(X_train, y_train)
#print and store the best result
print(f'Optimized Random Forest score: {search.best_score_} with parameters: {search.best_params_}.')
optimized_models['Optimized Random Forest'] = (search.best_score_, search.best_params_)
#make predictions
predictions = search.predict(X_test)
#Calculate Accuracy, Precision and Recall Metrics as well as Confusion Matrix
accuracy_opt['Optimized Random Forest'] = accuracy_score(predictions, y_test)
precision_opt['Optimized Random Forest'] = precision_score(predictions, y_test)
recall_opt['Optimized Random Forest'] = recall_score(predictions, y_test)
confusion_opt['Optimized Random Forest'] = confusion_matrix(y_test, predictions)


#tuning Naive Bayes
#no hyperparameters to tune


#tuning K-Nearest Neighbors
#define search space
params = dict()
params['n_neighbors'] = [1,3,5,10,20]
params['leaf_size'] = [10, 20, 30, 40]
params['p'] = [1,2,3]
params['weights'] = ['uniform', 'distance']
params['metric'] = ['minkowski', 'chebyshev', 'manhattan']
#define the search
search = GridSearchCV(KNeighborsClassifier(), params, scoring='accuracy', n_jobs=-1, cv=cv)
#perform the search
search.fit(X_train, y_train)
#print and store the best result
print(f'Optimized K-Nearest Neighbors score: {search.best_score_} with parameters: {search.best_params_}.')
optimized_models['Optimized K-Nearest Neighbors'] = (search.best_score_, search.best_params_)
#make predictions
predictions = search.predict(X_test)
#Calculate Accuracy, Precision and Recall Metrics as well as Confusion Matrix
accuracy_opt['Optimized K-Nearest Neighbors'] = accuracy_score(predictions, y_test)
precision_opt['Optimized K-Nearest Neighbors'] = precision_score(predictions, y_test)
recall_opt['Optimized K-Nearest Neighbors'] = recall_score(predictions, y_test)
confusion_opt['Optimized K-Nearest Neighbors'] = confusion_matrix(y_test, predictions)

#create dataframe with optimized results
df_model_opt = pd.DataFrame(index=optimized_models.keys(), columns=['Parameters', 'Accuracy', 'Precision', 'Recall', 'Confusion Matrix'])
df_model_opt['Parameters'] = [value[1] for value in optimized_models.values()]
df_model_opt['Accuracy'] = accuracy_opt.values()
df_model_opt['Precision'] = precision_opt.values()
df_model_opt['Recall'] = recall_opt.values()
df_model_opt['Confusion Matrix'] = confusion_opt.values()

#store results to csv (insert your filepath here)
df_model.to_csv([your filepath here], index=True)
df_model_opt.to_csv([your filepath here], index=True)