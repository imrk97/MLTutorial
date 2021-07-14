


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc, plot_roc_curve, precision_recall_curve
from xgboost import XGBClassifier

def roc_plot(classifier, X_test, y_test):
    probs = classifier.predict_proba(X_test)
    #a  = classifier.predict()
    preds = probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc}')
    plt.plot(fpr, tpr, label = 'AUC = %0.4f' %roc_auc, color = 'blue')
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1], 'r--')
    
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc
def pr_plot(classifier,X_test, y_test):
    y_probs = classifier.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    y_probs = y_probs[:, 1]
    # predict class values
    yhat = classifier.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, yhat)
    #lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    #print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    Class = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [Class, Class], linestyle='--', label='Malign')
    plt.plot(recall, precision, marker='.', label='Adaboost')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
df = pd.read_csv('C:/Users/rohan/Desktop/GIT/creditcard.csv')

#df["Time"] = df["Time"].apply(lambda x : x / 3600 % 24)
df.describe().T
df.info()
df['Class'].value_counts()
sns.heatmap(df.isnull(), cmap='Blues_r', yticklabels=False)
#sns.histplot(df[df['Class']==0])
sns.countplot(x = 'Class', data=df)
sns.histplot(data = df, x='Amount', hue='Class', kde=True)
sns.pairplot(df[['Time','Amount','Class']], hue = 'Class')
sns.displot(df[['Amount']])

sns.boxplot(y = 'Amount', x='Class', data = df)


corr = df.corr()

plt.figure(figsize = (10,8))
sns.heatmap(corr, cmap = "coolwarm", linewidth = 2, linecolor = "white")
plt.title("Correlation")
plt.show()
print(corr)

df['logAmount'] = np.log2(df['Amount']+1)
sns.displot(df[['logAmount']], bins = 20, kde = True)
sns.histplot(data = df, x='logAmount', hue='Class', kde=True)

sns.boxplot(x='Class', y='logAmount', data = df)
#df = df.drop(['Time', 'Amount'], axis = 1)

X = df.drop(['Class'],axis=1)
y = df['Class']

X_nm, y_nm = NearMiss().fit_resample(X, y)
print(X_nm.shape, y_nm.shape)

X = StandardScaler().fit_transform(X_nm)
y = y_nm.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33, random_state=0)
classifier = AdaBoostClassifier()
#classifier= XGBClassifier()
classifier.fit(X_train, y_train)
print(f'F1 Score: {f1_score(y_test, classifier.predict(X_test))}')
print(f'Accuracy: {accuracy_score(y_test, classifier.predict(X_test))}')
roc_plot(classifier, X_test, y_test)
pr_plot(classifier, X_test, y_test)
