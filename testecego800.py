import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import numpy as np


#-------------------------------------------------------------------------------
#carrega os dados a serem testados
df = pd.read_csv('env_vital_signals 800 teste cego.txt', header = None)

indices_desejados = [3, 4, 5, 7]
df_selecionado = df.iloc[:, indices_desejados]

df_selecionado.columns = ["qPA", "Pulso", "frequência Respiratória", "y"]

#print(df_selecionado)
#print(df_selecionado["y"].value_counts())

#divide entre os usados para o teste e a saida que sera usada para calcular os acertos
x = df_selecionado.drop("y", axis=1)
y = df_selecionado["y"]
#---------------------------------------------------------------------------------

#carrega a arvore criada na etapa de aprendizado

clf = joblib.load('modelo_arvore_decisao.pkl')
previsoes = clf.predict(x)

#print(previsoes)

previsoes = clf.predict(x)
valores_unicos, contagens = np.unique(previsoes, return_counts=True)

for valor, contagem in zip(valores_unicos, contagens):
    print(f"Previsões do tipo {valor}: {contagem}")

#---------------------------------------------------------------------------------

#calcula a proporção de previsões incorretas em relação ao total
previsoes = clf.predict(x)
erro = 1 - accuracy_score(y, previsoes)
acuracia = accuracy_score(y, previsoes)

print(f"Acuracia: {acuracia:.2f}")
print(f"Taxa de Erro: {erro:.2f}")

#calcula a média dos quadrados das diferenças entre as previsões e os valores reais
erro_quadratico_medio = mean_squared_error(y, previsoes)

print(f"Erro Quadrático Médio: {erro_quadratico_medio:.2f}")

#cria uma matriz de confusao
matriz_confusao = confusion_matrix(y, previsoes)
print("Matriz de Confusão:")
print(matriz_confusao)

