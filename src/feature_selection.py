from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    RFECV,
    SelectFromModel
)
from sklearn.linear_model import LogisticRegression


def get_feature_selector(method: str = "all", k: int = 8, estimator=None, step: int = 1, cv: int = 3, random_state: int = 42, n_features_to_select: int = None, **kwargs):
    if method is None or method == "all" or method == "none":
        return None

    method_lower = str(method).lower().strip()
    effective_k = n_features_to_select if n_features_to_select is not None else kwargs.get("n_features_to_select", kwargs.get("k", k))
    step = kwargs.get("step", step)

    if estimator is None:
        estimator = LogisticRegression(random_state=random_state, solver="liblinear")

    if method_lower in ("anova", "f_classif", "kbest", "selectkbest"):
        return SelectKBest(score_func=f_classif, k=effective_k)
    elif method_lower in ("mutual_info", "mutual_information", "mi"):
        return SelectKBest(score_func=mutual_info_classif, k=effective_k)
    elif method_lower in ("rfe",):
        return RFE(estimator=estimator, n_features_to_select=effective_k, step=step)
    elif method_lower in ("rfecv",):
        return RFECV(estimator=estimator, cv=cv, scoring="roc_auc", step=step)
    elif method_lower in ("model", "selectfrommodel"):
        threshold = kwargs.get("threshold", "mean")
        return SelectFromModel(estimator=estimator, threshold=threshold)
    else:
        # Graceful fallback: SelectKBest ANOVA
        return SelectKBest(score_func=f_classif, k=effective_k)
