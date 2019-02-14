# Hospital-Readmission-Prediction-Logistic-Regression-
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.feature_selection import RFE,RFECV
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
pd.options.display.max_columns = None 
pd.options.display.max_rows = None
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from collections import Counter


df = pd.read_csv('data_draft0.csv')


data_x000 = df.drop(['patient_nbr','readmitted'],axis=1)
data_y000 = df['readmitted']


X_train, X_test, y_train, y_test = train_test_split(data_x000,data_y000,test_size=0.25, random_state=21)


print('The number of samples into the train data is {}.'.format(X_train.shape[0]))


print('The number of samples into the test data is {}.'.format(X_test.shape[0]))


print('Patient readmition in training data (1 = Yes, 0 = No):')
print(y_train.value_counts())

#X_train.to_csv('x_train.csv',index=False)
#X_test.to_csv('x_test.csv',index=False)
#y_train.to_csv('y_train.csv',index=False)
#y_test.to_csv('y_test.csv',index=False)


ros = RandomOverSampler(random_state=0)
data_x001, data_y001 = ros.fit_resample(X_train, y_train)


Counter(data_y001)


data_x=pd.DataFrame(data_x001,columns=data_x000.columns)
data_y=pd.DataFrame({'readmitted':data_y001})['readmitted']


param_grid = {'C': np.arange(1.0e-05, 3, 0.1)}
scoring = {'recall':'recall','average_precision': 'average_precision', 'f1': 'f1','roc_auc':'roc_auc'}


gs = GridSearchCV(LogisticRegression(), return_train_score=True,param_grid=param_grid,scoring=scoring, cv=10,refit='roc_auc')
gs.fit(data_x, data_y)


print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)


results = gs.cv_results_


rfecv = RFECV(estimator=LogisticRegression(C=gs.best_params_['C']), cv=10, scoring='roc_auc')
rfecv.fit(data_x, data_y)


print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(data_x.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


selected_feature = data_x.columns[rfecv.support_]


logreg = LogisticRegression(C=gs.best_params_['C'])
logreg.fit(data_x[selected_feature],data_y)
pred_readmission=logreg.predict(X_test[selected_feature])


from sklearn.metrics import confusion_matrix

confmat = confusion_matrix(y_true= y_test, y_pred= pred_readmission)
print(confmat)

# Graph the confusion matrix
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


from sklearn.metrics import precision_score, recall_score, accuracy_score,classification_report,roc_curve,auc

print('Recall: %.3f' % recall_score(y_true= y_test, y_pred= pred_readmission))
print('accuracy: %.3f' % accuracy_score(y_test, pred_readmission))
print('precision: %.3f' % precision_score(y_test, pred_readmission))
print(classification_report(y_test, pred_readmission))


[fpr, tpr, thr] = roc_curve(y_test, pred_readmission)
idx = np.min(np.where(tpr > 0.95))
plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Readmition (ROC) curve')
plt.legend(loc="lower right")
plt.show()


#pd.DataFrame(pred_readmission).to_csv('log_pred.csv')
