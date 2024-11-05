import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

df = pd.read_csv("creditcard.csv")
X = df.iloc[:, 0:30]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sm = SMOTE(random_state=42)


dtc = RandomForestClassifier()

dtc_params = RandomForestClassifier(criterion="log_loss", max_features="sqrt")

X_test_smote, y_test_smote = sm.fit_resample(X_test, y_test)

dtc.fit(X_train, y_train)

dtc_params.fit(X_train, y_train)

resp = dtc.predict(X_test)

resp_SMOTE = dtc.predict(X_test_smote)

resp_dtc_params = dtc_params.predict(X_test)

knn = KNeighborsClassifier()
svc = SVC()
vtc = VotingClassifier(estimators=[('dtc', dtc), ('knn', knn), ('svc', svc)], voting="hard")

vtc.fit(X_train, y_train)
resp_vtc = vtc.predict(X_test)

print("*"*30)

print("Padrão: ")
print(f"->Taxa de acerto: {accuracy_score(y_test, resp)}")
print(f"->Report: {classification_report(y_test, resp)}")
print(f"->Matriz: {confusion_matrix(y_test, resp)}")

print("*"*30)

print("Com SMOTE: ")
print(f"->Taxa de acerto: {accuracy_score(y_test_smote, resp_SMOTE)}")
print(f"->Report: {classification_report(y_test_smote, resp_SMOTE)}")
print(f"->Matriz: {confusion_matrix(y_test_smote, resp_SMOTE)}")

print("*"*30)

print("Com Params: ")
print(f"->Taxa de acerto: {accuracy_score(y_test, resp_dtc_params)}")
print(f"->Report: {classification_report(y_test, resp_dtc_params)}")
print(f"->Matriz: {confusion_matrix(y_test, resp_dtc_params)}")

print("*"*30)

print("Voting Classifier: ")
print(f"->Taxa de acerto: {accuracy_score(y_test, resp_vtc)}")
print(f"->Report: {classification_report(y_test, resp_vtc)}")
print(f"->Matriz: {confusion_matrix(y_test, resp_vtc)}")

print("*"*30)

print("Com base em análises, o melhor continuou sendo o padrão º-º)b")
