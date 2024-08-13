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

"""
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
"""
df = pd.read_csv('env_vital_signals.txt', header = None)

indices_desejados = [3, 4, 5, 7]
df_selecionado = df.iloc[:, indices_desejados]

#nova_linha = pd.DataFrame([["qPA", "Pulso", "frequência Respiratória", "y"]], columns=df_selecionado.columns)

# Concatene a nova linha com o DataFrame existente
#df_novo = pd.concat([nova_linha, df_selecionado], ignore_index=True)

df_selecionado.columns = ["qPA", "Pulso", "frequência Respiratória", "y"]

print(df_selecionado)

print(df_selecionado["y"].value_counts())

x = df_selecionado.drop("y", axis=1)
y = df_selecionado["y"]

#trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
print(f"Tamanho total do dataset: {len(x)}\n")
print(f"Dados de treinamento X ({len(X_train)}):\n{X_train[:3]} ...")
print(f"Dados de treinamento y:({len(y_train)})\n {y_train[:3]} ...")
print("---")
print(f"Dados de teste   X ({len(X_test)}):\n{X_test[:3]} ...")
print(f"Dados de teste   y:({len(y_test)})\n {y_test[:3]} ...")


# Parameters' definition
parameters = {
    'criterion': ['entropy'],
    'max_depth': [4, 6, 80],
    'min_samples_leaf': [2, 3, 5, 10]
}

# instantiate model
# random_state = 42 to be deterministic
model = DecisionTreeClassifier(random_state=42)

# grid search using cross-validation
# cv = 3 is the number of folds
# scoring = 'f' the metric for chosing the best model

"""

#mudei o scoring, antes não estava definido para multiclasses [0,1,2,3]

"""
clf = GridSearchCV(model, parameters, cv=30, scoring='f1_macro', verbose=4, error_score= 'raise')
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

import joblib

joblib.dump(best, 'modelo_arvore_decisao.pkl')