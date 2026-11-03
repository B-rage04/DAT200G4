"""
Brage Bestvold and Lindholm


The best model was printed out in the jupyter notebook when iterating through each classifier and seed

Results:
Best model: RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=5) 
Validation Score: 0.7717131474103586 
Train score: 0.8606164383561644
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

 
train = pd.read_csv("CA03_Dat200/train.csv")
X = train.drop(axis=1, labels=["quality", "Id"])
y = train["quality"]

test = pd.read_csv("CA03_Dat200/test.csv")
test = test.drop(axis=1, labels=["Id"])


best_model = RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=5)
best_model.fit(X, y)

y_pred = best_model.predict(test)

submission = pd.DataFrame()
submission["Id"] = range(len(y_pred))
submission["quality"] = y_pred

