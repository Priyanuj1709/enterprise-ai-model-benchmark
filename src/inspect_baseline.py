"""Show the words/phrases that push the model most strongly towards each category."""
import joblib
import numpy as np

model = joblib.load("models/tfidf_logreg.joblib")
vec = model.named_steps["tfidfvectorizer"]
clf = model.named_steps["logisticregression"]
names = vec.get_feature_names_out()

for i, label in enumerate(clf.classes_):
    top = np.argsort(clf.coef_[i])[::-1][:12]
    print(f"{label:16} {', '.join(names[top])}\n")