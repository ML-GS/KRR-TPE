import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, anneal, partial, rand
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
import warnings

warnings.filterwarnings("ignore")

method = 'KRR'
species = 'pine'
np.random.seed(123)
max_evals = 500
k_cv = 5
repeat_num = 5
trait_num = 20
traits = {
    'pine':np.array(range(15)),
    '1287':np.array(range(20)),
    'pig':np.array([0,1,2,3,4]),
    'cow':np.array([0,1]),
    'qtlmas':np.array([0,1,2]),
    'qtlmas16':np.array([0,1,2]),
    'rice':np.array([0,1,2]),

}



snp = pd.read_pickle("data/snp_%s" % species)
snp = PCA(snp.shape[0]).fit_transform(snp)
snp = pd.DataFrame(snp)
phes = pd.read_pickle("data/phe_%s" % species)


def pearson_r(y_pre, y_true):
    acc = pearsonr(y_pre, y_true)[0]
    return acc


my_score = make_scorer(pearson_r, greater_is_better=True)


def featureSelection_Kernel_RidgeRegression(k, kernel, alpha, degree, gamma):
    def select_pca(a, b):
        p = list(np.arange(a.shape[1]))
        p.reverse()
        return np.array(p), np.array(p)
    anova_filter = SelectKBest(select_pca, k=int(k))
    re = KernelRidge(kernel=kernel, alpha=alpha, degree=degree, gamma=gamma)
    fkr = make_pipeline(anova_filter, re)
    return fkr


def hyperopt_train_test(params):
    global ITERATION
    fkr = featureSelection_Kernel_RidgeRegression(**params)
    res = []
    for rep in range(repeat_num):
        kf = KFold(n_splits=k_cv, random_state=rep, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]
            fkr.fit(X_train, y_train)
            y_pred = fkr.predict(X_test)
            res.append(pearsonr(y_test, y_pred)[0])
    res_accs.iloc[ITERATION, :] = res
    ITERATION += 1
    return np.array(res).mean()


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

species = 'pine'
for trait in traits[species]:
    ITERATION = 0
    res_accs = pd.DataFrame(np.zeros([max_evals, k_cv*repeat_num]))
    print('Trait: ', trait)
    stds = []
    X = snp
    y = phes.iloc[:, trait]
    nona_index = np.where(y > -float('inf'))
    X = X.iloc[nona_index[0], :]
    y = y[nona_index[0]]
    X = np.array(X)
    y = np.array(y)

    dict(boosting_type=hp.choice('boosting_type', [
        {'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0, 1)},
        {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0, 1)},
        {'boosting_type': 'goss', 'subsample': 1.0}]),
         num_leaves=hp.quniform('num_leaves', 30, 150, 1),
         learning_rate=hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
         min_child_samples=hp.quniform('min_child_samples', 20, 200, 2),
         reg_alpha=hp.uniform('reg_alpha', 0, 2.0), reg_lambda=hp.uniform('reg_lambda', 0, 2.0),
         colsample_bytree=hp.uniform('colsample_by_tree', 0, 1.0),
         k=hp.choice('k', range(100, 10000, 10)),
         n_estimators=hp.choice('n_estimators', range(10, 100, 2))
         )
    space = {

        "k": hp.choice('k', range(1, X.shape[1])),
        'kernel': hp.choice('kernel', ["cosine", 'linear', 'rbf', 'poly']),
        'degree': hp.choice('degree', [1, 2, 3, 4]),
        "alpha": hp.choice("alpha", np.linspace(0,10,1000)),
        "gamma": hp.choice("gamma", [0.000001, 0.000005,0.00001,0.00005,0.0001,0.0005,0.001]),
    }
    trials = Trials()
    # best = fmin(f, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=True)
    best = fmin(f, space, algo=rand.suggest, max_evals=max_evals, trials=trials, verbose=True)
    ys = [-t['result']['loss'] for t in trials.trials]
    res_accs.to_excel('output/%s/RS_accs_10_%s_%s.xlsx' % (species, method, trait))
    # pd.DataFrame(trials.trials).to_excel('output/%s/RS_trials_10_%s_%s.xlsx' % (species, method, trait))
    print(np.array(ys).max())
    print('best:', best)

