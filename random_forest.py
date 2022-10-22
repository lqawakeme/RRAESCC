import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold

train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')
target = '0_1'
print(train_data[target].value_counts(), validation_data[target].value_counts())
train_result = train_data[target]
validation_result = validation_data[target]


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
    plt.savefig('machine_learning/AUC/'+figname + str(num) + '.jpg')
    plt.close()


def random_forest_01():
    x_columns = [x for x in train_data.columns if x not in [target, 'ID', 'cohort','path_origin_image','path_mask_image']]
    X_train = train_data[x_columns]
    Y_train = train_result

    x_validation_columns = [x for x in validation_data.columns if x not in [target, 'ID', 'cohort']]
    X_validation = validation_data[x_columns]
    Y_validation = validation_result

    rf0 = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=1000)
    parameters = {'max_features': 10}
    rf0.fit(X_train, Y_train)
    print(rf0.oob_score_)

    val_predict = rf0.predict_proba(X_validation)[:, 1]
    tra_predict = rf0.predict_proba(X_train)[:, 1]
    data_predict = pd.DataFrame({'pre': tra_predict, 'tru': Y_train})
    data_predict.to_csv('predict.csv')
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_train, tra_predict))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(Y_validation, val_predict))

    feat_important = rf0.feature_importances_
    # 特征名
    feat_name = train_data[x_columns].columns.values
    pds = pd.DataFrame({'feature': feat_name, 'importance': feat_important})
    pds.sort_values(by='importance')
    pds.to_csv('feature_importance_2.csv')
    print(len(feat_important), len(feat_name))
    plt.figure(figsize=(40, 40), dpi=80)
    plt.barh(range(len(feat_name)), feat_important, tick_label=feat_name)
    plt.savefig('fe_im.jpg')
    plt.close()
def draw_barh():
    cc = pd.read_csv('feature_importance.csv')
    feat_name = cc['feature'].head(10)
    feat_important = cc['importance'].head(10)
    plt.figure(figsize=(10, 10), dpi=300)
    plt.barh(range(len(feat_name)), feat_important, tick_label=feat_name)
    plt.savefig('fe_im.jpg')
    plt.close()

def random_forest_02():
    data10 = pd.read_csv('feature_importance.csv')
    important10 = data10['feature'].head(4).values
    X_train = train_data[important10]
    Y_train = train_result

    x_validation_columns = [x for x in validation_data.columns if x not in [target, 'ID', 'cohort']]
    X_validation = validation_data[important10]
    Y_validation = validation_result

    rf0 = RandomForestClassifier(oob_score=True, random_state=num, max_depth=3
                                 , criterion='entropy', n_estimators=100,
                                 max_features='log2')
    rf0.fit(X_train, Y_train)

    # val_predict = rf0.predict_proba(X_validation)[:, 1]
    # tra_predict = rf0.predict_proba(X_train)[:, 1]
    val_predict = rf0.predict(X_validation)
    tra_predict = rf0.predict(X_train)
    a = metrics.roc_auc_score(Y_train, tra_predict)
    b = metrics.roc_auc_score(Y_validation, val_predict)
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_train, tra_predict))
    # print("AUC Score (Test): %f" % metrics.roc_auc_score(Y_validation, val_predict))
    # print("ACC Score (Train): %f" % metrics.accuracy_score(Y_train, tra_predict))
    # print("ACC Score (Test): %f" % metrics.accuracy_score(Y_validation, val_predict))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_train, tra_predict))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(Y_validation, val_predict))
    auc_curve(Y_train, tra_predict, figname='Primary_AUC', num=num)
    auc_curve(Y_validation, val_predict, figname='Validation_AUC', num=num)
    joblib.dump(rf0, 'machine_learning/model/train_model'+str(num)+'.m')
    feat_important = rf0.feature_importances_
    # 特征名
    feat_name = train_data[important10].columns.values
    plt.figure(figsize=(10, 10), dpi=80)
    plt.barh(range(len(feat_name)), feat_important, tick_label=feat_name)
    plt.savefig('machine_learning/importance/fe_im_'+str(num)+'.jpg')

random_forest_02()
draw_barh()
