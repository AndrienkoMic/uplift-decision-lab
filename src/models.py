"""Fixed-complexity CatBoost S/T learners and a conversion propensity baseline."""

from catboost import CatBoostClassifier
from sklift.models import SoloModel, TwoModels

from src.features import CATEGORICAL


def estimator(seed, iterations):
    return CatBoostClassifier(
        iterations=iterations,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
        cat_features=CATEGORICAL,
    )


def fit_models(x, y, treatment, seed=42, iterations=250):
    # No class weighting: it would distort probabilities used in treatment differences.
    s = SoloModel(estimator(seed, iterations), method="dummy").fit(x, y, treatment)
    t = TwoModels(
        estimator_trmnt=estimator(seed, iterations),
        estimator_ctrl=estimator(seed + 1, iterations),
        method="vanilla",
    ).fit(x, y, treatment)
    response = estimator(seed + 2, iterations)
    response.fit(x.loc[treatment == 0], y.loc[treatment == 0])
    return {"s_learner": s, "t_learner": t, "response_baseline": response}


def predict_scores(models, x):
    return {
        name: model.predict_proba(x)[:, 1] if name == "response_baseline" else model.predict(x)
        for name, model in models.items()
    }
