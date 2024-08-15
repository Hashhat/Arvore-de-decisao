import pandas as pd
#import numpy as np
#from scipy import stats
#import matplotlib.pyplot as plt
#import seaborn as sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#---------------------------------------------------------------------------------
df = pd.read_csv('env_vital_signals.txt', header = None)

indices_desejados = [3, 4, 5, 7]
df_selecionado = df.iloc[:, indices_desejados]

df_selecionado.columns = ["qPA", "Pulso", "frequência Respiratória", "y"]

print(df_selecionado)

print(df_selecionado["y"].value_counts())

x = df_selecionado.drop("y", axis=1)
y = df_selecionado["y"]

#---------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
#print(f"Tamanho total do dataset: {len(x)}\n")
#print(f"Dados de treinamento X ({len(X_train)}):\n{X_train[:3]} ...")
#print(f"Dados de treinamento y:({len(y_train)})\n {y_train[:3]} ...")
#print("---")
#print(f"Dados de teste   X ({len(X_test)}):\n{X_test[:3]} ...")
#print(f"Dados de teste   y:({len(y_test)})\n {y_test[:3]} ...")


# Parameters' definition
parameters = {
    'criterion': ['entropy'],
    'max_depth': [4, 6, 80],
    'min_samples_leaf': [2, 3, 5, 10]
}

# instantiate model
# random_state = 42 to be deterministic
model = DecisionTreeClassifier(random_state=42)
"""
#mudei o scoring, antes não estava definido para multiclasses [1,2,3,4]
"""
# grid search using cross-validation
#cv=define diretamente o de divisoes que cada combinaçao entre max_depth e min_samples_leaf terão
#f1_macro = fornece uma visão equilibrada do desempenho do modelo, dando igual importância a cada classe, independentemente do número de exemplos em cada classe 

clf = GridSearchCV(model, parameters, cv=30, scoring='f1_macro', verbose=4, error_score= 'raise')
clf.fit(X_train, y_train)

# Melhor arvore usando o f1_macro
best = clf.best_estimator_
print("\n* Melhor classificador *")
print(clf.best_estimator_)

print("Pontuações para cada combinação de parâmetros:")
for mean_score, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
    print(f"Parâmetros: {params} - Pontuação média: {mean_score:.4f}")


#---------------------------------------------------------------------------------
# Predicoes com dados do treinamento
y_pred_train = best.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train) * 100
print(f"Acuracia com dados de treino: {acc_train:.2f}%")

# com dados de teste (nao utilizados no treinamento/validacao)
y_pred_test = best.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test) * 100
print(f"Acuracia com dados de teste: {acc_test:.2f}%")

#cria uma matriz de confusao(dados de teste)
matriz_confusao = confusion_matrix(y_test, y_pred_test)
print("Matriz de Confusão:")
print(matriz_confusao)

#---------------------------------------------------------------------------------
"""
pode ser bom, ler mais sobre
"""
# Obtendo o relatório de classificação
relatorio = classification_report(y_test, y_pred_test, target_names=['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3'])

print(relatorio)
#---------------------------------------------------------------------------------

"""
#Cria uma imagem da arvore
from sklearn import tree
fig = plt.figure(figsize=(200, 200))
tree.plot_tree(best, feature_names=["X0", "X1" , "x2"], filled=False, rounded=False, class_names=["0", "1", "2", "3"], fontsize=5)
plt.savefig('grafico.png', format='png')
plt.show()
"""

#---------------------------------------------------------------------------------
#cria o arquivo da arvore a ser exportada
import joblib
joblib.dump(best, 'modelo_arvore_decisao.pkl')