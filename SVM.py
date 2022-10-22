import datetime

starttime = datetime.datetime.now()
from PIL import Image
import xlrd
import joblib
import numpy as np
import pandas as pd
import os
import cv2
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import svm,linear_model
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import StratifiedKFold

def auc_score(pro_y, pro_scores):
    fpr, tpr, threshold = roc_curve(pro_y, pro_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def auc_curve(pro_y, pro_scores, figname, num):
    fpr, tpr, threshold = roc_curve(pro_y, pro_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(figname)
    plt.legend(loc="lower right")
    plt.savefig('augmentation_image/paper_images/SVM_LR/' + figname + str(num) + '.png')
    plt.close()
def bootstrap_auc(y, pred, bootstraps=100, fold_size=840):
    global max_, min_, mean
    statistics = []

    df = pd.DataFrame(columns=['y', 'pred'])
    df.loc[:, 'y'] = y
    df.loc[:, 'pred'] = pred
    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]
    prevalence = len(df_pos) / len(df)
    for i in range(bootstraps):
        pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
        neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

        y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
        pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
        score = roc_auc_score(y_sample, pred_sample)
        statistics.append(score)

    statistics.sort()
    mean = np.mean(statistics)
    std = np.std(statistics)
    min_ = statistics[3]
    max_ = statistics[97]

    return max_,min_,mean


primary_CI =[]
primary_mean = []
validation_CI = []
validation_mean = []
train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')
target = '0_1'
print(train_data[target].value_counts(), validation_data[target].value_counts())
train_result = train_data[target]
validation_result = validation_data[target]
data10 = pd.read_csv('feature_importance.csv')
important10 = data10['feature'].head(5).values
X_train = train_data[important10]
Y_train = train_result
x_validation_columns = [x for x in validation_data.columns if x not in [target, 'ID', 'cohort']]
X_validation = validation_data[important10]
Y_validation = validation_result


classifier = svm.SVC(kernel='rbf', gamma=0.001, C=100, probability=True, random_state=None,class_weight='balanced')

classifier.fit(X_train,Y_train)

tra_predict = classifier.predict(X_train)
val_predict = classifier.predict(X_validation)
auc_curve(Y_train, tra_predict, figname='Primary_SVM(rbf)_AUC', num=0)
auc_curve(Y_validation, val_predict, figname='Internal_SVM(rbf)_AUC', num=0)
max1,min1,mean1 = bootstrap_auc(Y_train,tra_predict)
max2,min2,mean2 = bootstrap_auc(Y_validation,val_predict)
primary_CI.append([min1,max1])
primary_mean.append(mean1)
validation_CI.append(([min2,max2]))
validation_mean.append(mean2)
joblib.dump(classifier, 'machine_learning/SVM_linear/SVM_rbf_model'+str(0)+'.m')


classifier_1 = svm.SVC(kernel='linear', gamma=0.1, C=100, probability=True, random_state=0,class_weight='balanced')
classifier_1.fit(X_train,Y_train)
tra_predict1 = classifier_1.predict(X_train)
val_predict1 = classifier_1.predict(X_validation)
auc_curve(Y_train, tra_predict1, figname='Primary_SVM(linear)_AUC', num=0)
auc_curve(Y_validation, val_predict1, figname='Internal_SVM(linear)_AUC', num=0)
max1,min1,mean1 = bootstrap_auc(Y_train,tra_predict1)
max2,min2,mean2 = bootstrap_auc(Y_validation,val_predict1)
primary_CI.append([min1,max1])
primary_mean.append(mean1)
validation_CI.append(([min2,max2]))
validation_mean.append(mean2)
joblib.dump(classifier_1, 'machine_learning/SVM_linear/SVM_linear_model'+str(0)+'.m')


classifier_l = linear_model.LinearRegression()
classifier_l.fit(X_train,Y_train)
tra_predict2 = classifier_l.predict(X_train)
val_predict2 = classifier_l.predict(X_validation)
auc_curve(Y_train, tra_predict2, figname='Primary_LR_AUC', num=0)
auc_curve(Y_validation, val_predict2, figname='Internal_LR_AUC', num=0)
max1,min1,mean1 = bootstrap_auc(Y_train,tra_predict2)
max2,min2,mean2 = bootstrap_auc(Y_validation,val_predict2)
primary_CI.append([min1,max1])
primary_mean.append(mean1)
validation_CI.append(([min2,max2]))
validation_mean.append(mean2)
joblib.dump(classifier_l, 'machine_learning/SVM_linear/LR_model'+str(0)+'.m')

df = pd.DataFrame()
df['Primary_CI'] = primary_CI
df['Primary_Mean'] = primary_mean
df['validation_CI'] = validation_CI
df['Validation_Mean'] = validation_mean

df.to_csv("machine_learning/SVM_linear/CI.csv")




