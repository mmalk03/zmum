import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_feature_scatter(df1, df2, features):
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4, 4, figsize=(14, 14))

    i = 0
    for feature in features:
        i += 1
        plt.subplot(4, 4, i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show()


def plot_feature_distribution(df1, df2, label1, label2, features):
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5, 5, figsize=(18, 18))

    i = 0
    for feature in features:
        i += 1
        plt.subplot(5, 5, i)
        sns.kdeplot(df1[feature], bw=0.5, label=label1)
        sns.kdeplot(df2[feature], bw=0.5, label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show()


def plot_new_feature_distribution(df1, df2, label1, label2, features):
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2, 4, figsize=(18, 22))

    i = 0
    for feature in features:
        i += 1
        plt.subplot(2, 4, i)
        sns.kdeplot(df1[feature], bw=0.5, label=label1)
        sns.kdeplot(df2[feature], bw=0.5, label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show()


def plot_roc_curve(results):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    for (model_name, y, y_pred) in results:
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc = round(roc_auc_score(y, y_pred), 4)
        plt.plot(fpr, tpr, label=f"{model_name} AUC: {auc}")
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig('roc-comparison.png')
