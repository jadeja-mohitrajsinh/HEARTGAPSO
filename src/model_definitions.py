from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_models(random_state: int = 42) -> dict:
    models = {
        'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state, max_depth=5),
        'SVM': CalibratedClassifierCV(SVC(random_state=random_state), ensemble=False),
        'RandomForest': RandomForestClassifier(random_state=random_state, n_estimators=100),
        'ExtraTrees': ExtraTreesClassifier(random_state=random_state, n_estimators=100),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(random_state=random_state, eval_metric='logloss')
    else:
        models['XGBoost'] = GradientBoostingClassifier(random_state=random_state, n_estimators=100)
    return models


def get_stacked_ensemble(base_models: dict | None = None, meta_learner=None, random_state: int = 42) -> StackingClassifier:
    if base_models is None:
        base_models = get_models(random_state=random_state)
    if meta_learner is None:
        meta_learner = LogisticRegression(random_state=random_state, solver='liblinear')
        
    xgb_clf = base_models.get('XGBoost')
    if xgb_clf is None:
        if HAS_XGBOOST:
            xgb_clf = XGBClassifier(random_state=random_state, eval_metric='logloss')
        else:
            xgb_clf = GradientBoostingClassifier(random_state=random_state, n_estimators=100)
            
    estimators = [
        ('svm', base_models.get('SVM', CalibratedClassifierCV(SVC(random_state=random_state), ensemble=False))),
        ('xgb', xgb_clf),
        ('et', base_models.get('ExtraTrees', ExtraTreesClassifier(random_state=random_state, n_estimators=100)))
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )
