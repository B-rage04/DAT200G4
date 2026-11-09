"""
CA4 by Lasse Lindholm and Brage Bestvold Group 4 on Kaggle

The best estimator found by the pipelines in the jupyter notebook was a randomforest classifier with the parameters:
class weight: balanced, max depth: 20, min samples leaf: 3
"""


import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

train = pd.read_csv('CA4_Dat200/trainCA4DAT200.csv')
test = pd.read_csv('CA4_Dat200/testCA4DAT200.csv')

X = train.drop(columns=["CustomerId", "Churn", "DayMinutes", "EveMinutes", "NightMinutes", "IntlMinutes"])
y = train["Churn"]

categorical_data = ["State","AreaCode", "InternationalPlan", "VoiceMailPlan"]

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_data),
        ('numerical', StandardScaler(), X.drop(columns=categorical_data).columns.values)
    ]
)


pipe_rf = make_pipeline(preprocessor,
                            RandomForestClassifier(n_estimators=500, random_state=1))

cross_validation = StratifiedKFold(10, shuffle=True, random_state=1)

rf_grid = { "randomforestclassifier__max_depth": [None, 5, 10, 15, 20, 25, 30],
            "randomforestclassifier__min_samples_leaf": [1, 3, 5],
            "randomforestclassifier__class_weight": [None, "balanced"]}


rf_gs = GridSearchCV(estimator=pipe_rf,   
                    param_grid=rf_grid,
                    scoring='f1_macro',
                    cv=cross_validation,
                    n_jobs=-1)


rf_gs = rf_gs.fit(X, y)

test_customerId = test['CustomerId']
X_test = test.drop(columns='CustomerId')

rf_gs.best_estimator_.fit(X, y)
y_pred = rf_gs.best_estimator_.predict(X_test)

submission = pd.DataFrame({'CustomerId': test_customerId,'Churn': y_pred})

submission.to_csv("CA4_Dat200/submission.csv", index=False)