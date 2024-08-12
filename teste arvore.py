import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#import seaborn as sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#with open('C:\\Users\\r\\Desktop\\Arvore de decisao\\env_vital_signals.txt', 'r') as arquivo:
    #conteudo = arquivo.read()
    #print(conteudo)


dados = []
with open('C:\\Users\\r\\Desktop\\Arvore de decisao\\env_vital_signals.txt', 'r') as arquivo:
    for linha in arquivo:
        partes = linha.strip().split(',')  # `strip()` remove a quebra de linha, `split(',')` separa por vírgula
        dados.append(partes)

#dados[linha][coluna]

#print(dados[1][1])

Xm = []
Ym = []
for dado in dados:
    Xm.append([float(dado[3]), float(dado[4]), float(dado[5])])
    Ym.append(int(dado[7]))

#print(Xm) 

dfx = pd.DataFrame(Xm, columns=['qPA', 'pulso', 'frequência respiratória'])

dfy = pd.DataFrame(Ym, columns=['classes de gravidade'])


X_train, X_test, y_train, y_test = train_test_split(Xm, Ym, test_size=0.25, shuffle=True)
print(f"Tamanho total do dataset: {len(Xm)}\n")
print(f"Dados de treinamento X ({len(X_train)}):\n{X_train[:3]} ...")
print(f"Dados de treinamento y:({len(y_train)})\n {y_train[:3]} ...")
print("---")
print(f"Dados de teste   X ({len(X_test)}):\n{X_test[:3]} ...")
print(f"Dados de teste   y:({len(y_test)})\n {y_test[:3]} ...")


# Parameters' definition
parameters = {
    'criterion': ['entropy'],
    'max_depth': [6, 8],
    'min_samples_leaf': [2, 3, 4]
}

# instantiate model
# random_state = 42 to be deterministic
model = DecisionTreeClassifier(random_state=42)

# grid search using cross-validation
# cv = 3 is the number of folds
# scoring = 'f' the metric for chosing the best model
clf = GridSearchCV(model, parameters, cv=3, scoring='f1', verbose=4)
clf.fit(X_train, y_train)

# the best tree according to the f1 score
best = clf.best_estimator_
print("\n* Melhor classificador *")
print(clf.best_estimator_)

# Predicoes
# com dados do treinamento
y_pred_train = best.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train) * 100
print(f"Acuracia com dados de treino: {acc_train:.2f}%")

# com dados de teste (nao utilizados no treinamento/validacao)
y_pred_test = best.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test) * 100
print(f"Acuracia com dados de teste: {acc_test:.2f}%")

from sklearn import tree
fig = plt.figure(figsize=(200, 200))
tree.plot_tree(best, feature_names=["X0", "X1" , "x2"], filled=False, rounded=False, class_names=["0", "1", "2", "3"], fontsize=5)
plt.savefig('grafico.png', format='png')
plt.show()


# Matriz de confusão
#from sklearn.metrics import ConfusionMatrixDisplay
#ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
#print(classification_report(y_test, y_pred_test))