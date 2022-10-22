from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import numpy as np
import xlrd
import os, shutil
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn.calibration as cal
import joblib
import random


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
    #plt.title(figname)
    plt.legend(loc="lower right")
    plt.savefig('augmentation_image/paper_images/AUC/RF_AUCs/' + figname + str(num) + '.tif', dpi=1200)
    plt.close()


def auc_curve_all(names,data, figname, num):
    colors = [(219/255,49/255,36/255),(144/255,201/255,230/255),(39/255,158/255,188/255),(18/255,104/255,131/255),(2/255,48/255,71/255),
             (255/255,183/255,3/255),(253/255,159/255,2/255),(251/255,132/255,2/255),(203/255,153/255,126/255),(84/255,104/255,111/255),(1/255,53/255,101/255)]
    lw = 3
    plt.figure(figsize=(10, 10))
    fpr, tpr, threshold = roc_curve(data[0][0], data[0][1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, color=colors[0],
             lw=lw, label='%s:%0.3f' % (names[0], roc_auc))
    i = 1
    for pro_y,pro_scores in data[1:]:
        fpr, tpr, threshold = roc_curve(pro_y, pro_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.plot(fpr, tpr, color=colors[i],
                 lw=1, label='%s:%0.3f' % (names[i],roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    #plt.title(figname)
    plt.legend(loc="lower right")
    plt.savefig('augmentation_image/paper_images/AUC/RF_AUCs/5_' + figname + str(num) + '.png')
    plt.close()

def draw_hist(names,data, figname, num):
    plt.figure()

def auc_curve_2(pro_y, pro_scores, p1, p2, figname, num):
    fpr, tpr, threshold = roc_curve(pro_y, pro_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fpr2, tpr2, threshold2 = roc_curve(p1, p2, pos_label=1)
    roc_auc2 = auc(fpr2, tpr2)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr2, tpr2, color='red',
             lw=lw, label='Att. pretrained Network ROC curve (area = %0.3f)' % roc_auc2)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='CNN ROC curve (area = %0.3f)' % roc_auc)

    ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity', fontdict={'fontsize': 15})
    plt.ylabel('Sensitivity', fontdict={'fontsize': 15})
    #plt.title(figname, fontdict={'fontsize': 15})
    plt.legend(loc="lower right")
    plt.savefig('augmentation_image/paper_images/AUC/' + figname + str(num) + '.tif', dpi=1200)
    plt.close()


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(y_pred_score, y_label, figname):
    # Plot
    thresh_group = np.arange(0, 1, 0.001)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(thresh_group, net_benefit_model, lw=lw, color='darkslategray', label='Model')
    plt.plot(thresh_group, net_benefit_all, lw=lw, color='peru', label='All patients respond to radiotherapy')
    plt.plot((0, 1), (0, 0), color='navy', lw=lw, linestyle=':', label='No patients respond to radiotherapy')
    # plt.plot((0,1),(1,1), color ='navy', lw = lw, linestyle =':')
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    #  plt.fill_between(thresh_group, y1, y2, color ='crimson', alpha =0.2)
    # Figure Configuration
    plt.xlim(0, 0.9)
    plt.ylim(net_benefit_model.min() - 0.03, net_benefit_model.max() + 0.05)  # adjustify the y axis limitation
    plt.xlabel(
        xlabel='Threshold Probability',
        fontdict={'fontsize': 15}
    )
    plt.ylabel(
        ylabel='Net Benefit',
        fontdict={'fontsize': 15}
    )
    plt.grid('major')
    # plt.spines['right'].set_color((0.8,0.8,0.8))
    # plt.spines['top'].set_color((0.8,0.8,0.8))
    plt.legend(loc='upper right')
    plt.savefig("augmentation_image/paper_images/DCA/" + figname + '15.tif', dpi=1200)


pretrain_primary = pd.read_csv("augmentation_image/patient_prediction_pretrained_primary.csv")
pretrain_validation = pd.read_csv("augmentation_image/patient_prediction_pretrained_validation.csv")
prediction_test = pd.read_csv("augmentation_image/patient_prediction_test.csv")
Y_pretrain = pretrain_primary['Y']
Y_pretrain_pred = pretrain_primary['predict_Y']
Y_val = pretrain_validation['Y_test']
Y_val_pred = pretrain_validation['predict_Y_test']
Y_external = prediction_test['Z_truth']
Y_test_CNN = prediction_test['predict_Z_cnn']
Y_test_pretrain = prediction_test['predict_Z_pretrain']


def draw_NN_AUC():
    table = pd.read_csv('models/division/patient_division397modify.csv')
    X = []
    Y = []
    X_test = []
    Y_test = []
    for m in range(table.shape[0]):
        id = table.iat[m, 0]
        cohort = table.iat[m, 2]
        k = table.iat[m, 1]
        for root, dirs, files in os.walk("/Users/liuqiang/PycharmProjects/Medical/CR-PR/final_final_image"):
            for dir in dirs:
                if str(dir).find(str(id), 0, len(str(dir))) > -1:
                    in_path = os.path.join(root, dir)
                    list = os.listdir(in_path)
                    for i in range(len(list)):
                        if str(list[i]).find('.DS_Store', 0, len(str(list[i]))) > -1:
                            continue
                        path = os.path.join(in_path, list[i])
                        if os.path.isfile(path):
                            im = Image.open(path)
                            im = im.resize((64, 64))
                            im = im.convert('RGB')
                            np_im = np.array(im) / 255.
                            if cohort == 'train':
                                X.append(np_im)
                                Y.append(k)
                            elif cohort == 'validation':
                                X_test.append(np_im)
                                Y_test.append(k)

    print(len(X), len(Y), len(X_test), len(Y_test))
    X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    model = load_model('models/model/my_model_397modify.h5')
    scores = model.predict(X)
    scores2 = model.predict(X_test)
    auc_curve_2(Y, scores, Y_pretrain, Y_pretrain_pred, figname='Radiomic signature in primary cohort', num=3)
    auc_curve_2(Y_test, scores2, Y_val, Y_val_pred, figname='Radiomic signature in internal cohort', num=3)
    auc_curve_2(Y_external, Y_test_CNN, Y_external, Y_test_pretrain, figname='Radiomic signature in external cohort',
                num=3)
    Y_all_pred = []
    Y_all = []
    for k in scores:
        for num in k:
            Y_all_pred.append(num)
    for k in scores2:
        for num in k:
            Y_all_pred.append(num)
    for num in Y:
        Y_all.append(num)
    for num in Y_test:
        Y_all.append(num)
    print(len(Y_all_pred), len(Y_all))
    plot_DCA(Y_all_pred, Y_all, 'Decision curve in CNN model')


def draw_DCA():
    print(len(Y_pretrain_pred), len(Y_val_pred), len(Y_test_pretrain))
    print(len(Y_pretrain), len(Y_val), len(Y_external))

    Y_all_pred = []
    Y_all = []
    for num in Y_pretrain_pred:
        Y_all_pred.append(num)
    for num in Y_val_pred:
        Y_all_pred.append(num)
    for num in Y_test_pretrain:
        Y_all_pred.append(num)
    for num in Y_pretrain:
        Y_all.append(num)
    for num in Y_val:
        Y_all.append(num)
    for num in Y_external:
        Y_all.append(num)
    print(len(Y_all_pred), len(Y_all))
    plot_DCA(Y_all_pred, Y_all, 'Decision curve in Att. pretrained model')

def calibration_curve():
    pretrain_primary_true, pretrain_primary_pred = cal.calibration_curve(Y_pretrain, Y_pretrain_pred, n_bins=10)
    pretrain_internal_true, pretrain_internal_pred = cal.calibration_curve(Y_val, Y_val_pred, n_bins=10)
    pretrain_external_true, pretrain_external_pred = cal.calibration_curve(Y_external, Y_test_pretrain, n_bins=10, normalize=True)

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(pretrain_primary_true, pretrain_primary_pred, color='darkorange',
             lw=lw, marker='o', label='Training cohort')
    # plt.plot(pretrain_external_true, pretrain_external_pred, color='red',
    #          lw=lw, marker='o', label='CNN')
    plt.plot(pretrain_internal_true, pretrain_internal_pred, color='navy',
             lw=lw, marker='o', label='Internal validation cohort')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',label='Prefect prediction')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Model predict probability',fontdict={'fontsize': 15})
    plt.ylabel('Actual pCR rate',fontdict={'fontsize': 15})
    #plt.title('Calibration of radiomic signature',fontdict={'fontsize': 15})
    plt.legend(loc="lower right")
    plt.savefig("augmentation_image/paper_images/CalibrationCurve/Calibration of radiomic signature.tif", dpi=1200)
    plt.savefig("augmentation_image/paper_images/CalibrationCurve/Calibration of radiomic signature.png")

def random_forest_AUC():
    model = joblib.load('machine_learning/model/train_model12.m')
    target = '0_1'
    train_data = pd.read_csv('train.csv')
    validation_data = pd.read_csv('validation.csv')
    train_result = train_data[target]
    validation_result = validation_data[target]
    data10 = pd.read_csv('feature_importance.csv')
    important10 = data10['feature'].head(5).values
    print(important10)
    X_train = train_data[important10]
    Y_train = train_result
    X_validation = validation_data[important10]
    Y_validation = validation_result
    scores2_ = model.predict_proba(X_validation)[:, 1]
    scores_ = model.predict_proba(X_train)[:, 1]
    rf_train_data = [[Y_train,scores_]]
    rf_val_data = [[Y_validation,scores2_]]
    names = ['multivariate model']
    for num_i in range(5):
        print(num_i)
        model = joblib.load('machine_learning/10AUC/models/train_model1.m')
        names.append(important10[num_i])
        print(names)
        X_train = train_data[important10[num_i]].values.reshape(-1, 1)
        X_validation = validation_data[important10[num_i]].values.reshape(-1, 1)
        model.fit(X_train, Y_train)
        scores2_final = model.predict_proba(X_validation)[:, 1]
        scores_final = model.predict_proba(X_train)[:, 1]
        rf_train_data.append([Y_train, scores_final])
        rf_val_data.append([Y_validation, scores2_final])
    auc_curve_all(names,rf_train_data,'Random Forest performance evaluation in primary cohort',0)
    auc_curve_all(names,rf_val_data,'Random Forest performance evaluation in internal cohort',0)

random_forest_AUC()



