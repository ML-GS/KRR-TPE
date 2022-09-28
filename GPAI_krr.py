from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, anneal, partial, rand
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer
import warnings

warnings.filterwarnings("ignore")

method = 'KRR'
species = 'cow'
np.random.seed(123)
max_evals = 200
k_cv = 5
repeat_num = 10
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


def featureSelection_Kernel_RidgeRegression( kernel, alpha, degree, gamma):
    re = KernelRidge(kernel=kernel, alpha=alpha, degree=degree, gamma=gamma)
    return re


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
    space = {
        'kernel': hp.choice('kernel', ["cosine", 'linear', 'rbf', 'poly']),
        'degree': hp.choice('degree', [1, 2, 3, 4]),
        "alpha": hp.uniform("alpha", 0.01,100),
        "gamma": hp.loguniform("gamma", -15,-6),
    }
    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=True, )
    # best = fmin(f, space, algo=rand.suggest, max_evals=max_evals, trials=trials, verbose=True)
    ys = [-t['result']['loss'] for t in trials.trials]
    # res_accs.to_excel('output/%s/613RS_accs_10_%s_%s.xlsx' % (species, method, trait))
    # pd.DataFrame(trials.trials).to_excel('output/%s/RS_trials_10_%s_%s.xlsx' % (species, method, trait))
    print(np.array(ys).max())
    print('best:', best)


