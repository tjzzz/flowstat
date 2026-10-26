"""传统机器学习模型基类与常用封装。

移植自 aibee logstat 的 metrics/models/base/base_models.py，已脱敏。
提供统一的训练/调参/预测/评估/保存接口（BaseClassifier），并内置 LR / LGBM / XGB /
RF / KNN / DT / SGD / AdaBoost 等 sklearn 系模型，以及可选的 DeepFM/NFFM（需 deepctr）。

约定：`create_model` 创建 estimator，`tune_model` 网格/贝叶斯调参，`model_fit` 训练，
`model_predict` 预测，`evaluate_model` 给出二分类评估，`save_result_model` 落盘。
"""
import os
import pickle
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn import metrics


def check_model_par_dict(model, user_model_par_dict):
    """过滤掉 estimator 不接受的参数，避免 set_params 报错。"""
    par_keys = model.get_params().keys()
    return {k: v for k, v in user_model_par_dict.items() if k in par_keys}


class BaseClassifier(object):
    """所有分类模型的基类。子类须实现 create_model，并可重写 model_fit。"""

    def __init__(self, par_dict=None):
        self.par_dict = par_dict or {}
        self.tune_grid = {}
        self.model_pars = {}

    def get_all_models(self):
        """返回当前所有已定义子类的名字。"""
        return [sc.__name__ for sc in BaseClassifier.__subclasses__()]

    def set_tune_grid(self, user_define_tune_par_dict):
        self.tune_grid = user_define_tune_par_dict

    def get_model_pars(self):
        return self.model.get_params()

    def set_model_pars(self, user_define_model_pars):
        self.model_pars = user_define_model_pars

    def create_model(self, user_model_par_dict=None):
        user_model_par_dict = user_model_par_dict or {}
        user_model_par_dict = check_model_par_dict(self.model, user_model_par_dict)
        self.model.set_params(**user_model_par_dict)

    def tune_model(self, trainX, trainY, user_define_tune_par_dict=None):
        """网格或贝叶斯调参。method=sklearn|tune-sklearn；search_optimization 用于 tune-sklearn。"""
        user_define_tune_par_dict = user_define_tune_par_dict or {}
        tune_grid = user_define_tune_par_dict.get('tune_par_dict', self.tune_grid) or self.tune_grid
        method = user_define_tune_par_dict.get('tune_method', 'sklearn')
        search_optimization = user_define_tune_par_dict.get('search_optimization', 'random')

        if method == 'tune-sklearn':
            from tune_sklearn import TuneSearchCV
            gsearch = TuneSearchCV(self.model, param_distributions=tune_grid, scoring='roc_auc',
                                   cv=3, search_optimization=search_optimization)
        else:
            gsearch = GridSearchCV(estimator=self.model, param_grid=tune_grid, scoring='roc_auc',
                                   cv=3, n_jobs=2, verbose=1)

        gsearch.fit(trainX, trainY)
        print('best params', gsearch.best_params_)
        self.model = gsearch.best_estimator_

    def model_fit(self, trainX, trainY):
        self.model.fit(trainX, trainY)

    def model_predict(self, testX):
        return self.model.predict(testX)

    def plot_model(self, performance):
        """树模型返回特征重要性（name->importance）。"""
        feature_names = self.par_dict.get('sparse_features', []) + self.par_dict.get('dense_features', [])
        if getattr(self.model, 'name', '') in ('lgb', 'xgb') and feature_names:
            importance_list = self.model.feature_importances_
            tmp_list = sorted(
                [(feature_names[i], float(importance_list[i])) for i in range(len(importance_list))],
                key=lambda x: x[1], reverse=True)
            return {'feature_importance': tmp_list}
        return {}

    def evaluate_model(self, real_list, pre_score_list, prefix=''):
        from metrics.metrics_ml.ml_evaluate import cal_model_performance_binary
        pred_list = [int(score > 0.5) for score in pre_score_list]
        return cal_model_performance_binary(real_list, pred_list, prefix)

    def save_result_model(self, model, save_model_name, model_type='ml'):
        if model_type == 'dl':
            from tensorflow.python.keras.models import save_model
            save_model(model, save_model_name + '.h5')
        else:
            pickle.dump(model, open(save_model_name + '.pkl', 'wb'))


class LGBMModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'lgb'
        self.tune_grid = {
            'n_estimators': [200, 300],
            'max_depth': [5, 6, 7],
            'learning_rate': [0.1, 0.01],
        }

    def create_model(self, user_model_par_dict=None):
        import lightgbm
        object_type = self.par_dict.get('objective', 'binary')
        self.model = lightgbm.LGBMRegressor(objective=object_type, num_leaves=50, learning_rate=0.1,
                                            n_estimators=500, max_depth=7)
        super().create_model(user_model_par_dict)

    def model_fit(self, trainX, trainY):
        self.model.fit(trainX, trainY, categorical_feature=self.par_dict.get('sparse_features', []))


class XGBMModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'xgb'
        self.tune_grid = {
            'n_estimators': [200, 300],
            'max_depth': [5, 6, 7, 8],
            'learning_rate': [0.1, 0.01],
            'scale_pos_weight': [1],
        }

    def create_model(self, user_model_par_dict=None):
        import xgboost
        self.model = xgboost.XGBClassifier(objective='binary:logistic', n_estimators=500)
        super().create_model(user_model_par_dict)


class RandomForestModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'rf'
        self.tune_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 6, 7]}

    def create_model(self, user_model_par_dict=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=300, max_depth=5)
        super().create_model(user_model_par_dict)


class KNeighborsModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'kn'
        self.tune_grid = {
            'n_neighbors': [3, 5, 8, 10],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
        }

    def create_model(self, user_model_par_dict=None):
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=3)
        super().create_model(user_model_par_dict)


class DecisionTreeModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'dt'
        self.tune_grid = {
            'max_depth': [5, 6, 7],
            'min_samples_leaf': [2, 3, 4, 5, 6],
            'min_samples_split': [2, 5, 7, 9, 10],
            'criterion': ['gini', 'entropy'],
        }

    def create_model(self, user_model_par_dict=None):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(max_depth=4, class_weight='balanced')
        super().create_model(user_model_par_dict)


class SGDClassifierModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'sgd'
        self.tune_grid = {
            'penalty': ['elasticnet', 'l2', 'l1'],
            'learning_rate': ['constant', 'invscaling', 'adaptive', 'optimal'],
        }

    def create_model(self, user_model_par_dict=None):
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', penalty='l2', max_iter=5)
        super().create_model(user_model_par_dict)


class AdaBoostModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'adaboost'
        self.tune_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'algorithm': ['SAMME', 'SAMME.R'],
        }

    def create_model(self, user_model_par_dict=None):
        from sklearn.ensemble import AdaBoostClassifier
        self.model = AdaBoostClassifier(n_estimators=200, learning_rate=0.01)
        super().create_model(user_model_par_dict)


class LogisticRegressionModel(BaseClassifier):
    def __init__(self, par_dict=None):
        super().__init__(par_dict)
        self.name = 'lr'
        self.tune_grid = {'penalty': ['l2', 'none'], 'C': [0, 0.1, 1, 10]}

    def create_model(self, user_model_par_dict=None):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        super().create_model(user_model_par_dict)


class FMModel(BaseClassifier):
    """DeepFM / NFFM（需安装 deepctr）。可选，非核心依赖。"""

    def create_model(self, user_model_par_dict=None):
        from deepctr.models import DeepFM, NFFM
        from deepctr.inputs import SparseFeat, DenseFeat

        par_dict = self.par_dict
        code_mapping_dict = par_dict['code_mapping_dict']
        sparse_features = par_dict['sparse_features']
        dense_features = par_dict['dense_features']
        feature_names = sparse_features + dense_features

        fixlen_feature_columns = [
            SparseFeat(feat, vocabulary_size=len(code_mapping_dict[feat]) + 1, embedding_dim=8)
            for feat in sparse_features
        ] + [DenseFeat(feat, 1) for feat in dense_features]

        default_pars = self.par_dict.get('default_pars', {})
        if self.name == 'nffm':
            model = NFFM(fixlen_feature_columns, fixlen_feature_columns, task='binary',
                         dnn_dropout=default_pars.get('dnn_dropout', 0.2),
                         l2_reg_linear=default_pars.get('l2_reg_linear', 0.1),
                         l1_reg_embedding=default_pars.get('l1_reg_embedding', 0.1))
        else:
            model = DeepFM(fixlen_feature_columns, fixlen_feature_columns, task='binary',
                           dnn_dropout=default_pars.get('dnn_dropout', 0.2),
                           l2_reg_linear=default_pars.get('l2_reg_linear', 0.1),
                           l1_reg_embedding=default_pars.get('l1_reg_embedding', 0.1))
        model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
        self.model = model
