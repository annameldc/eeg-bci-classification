# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import classification_report

#le os arquivos da pasta
pasta = r"C:\Users\annam\OneDrive\Área de Trabalho\PROJETOS\ClassificacaoEEG\DadosS001"
arquivos = []
for file in os.listdir(pasta):
    if file.endswith(".edf"):
        arquivos.append(os.path.join(pasta, file))
print(arquivos)

#abre os arquivos
raws = []
for arq in arquivos:
    raw = mne.io.read_raw_edf(arq, preload=True)
    raws.append(raw)
    
#Concatena os arquivos e gera o plot
raw = mne.concatenate_raws(raws)
raw.plot()

#filtra
raw.filter(8.,30.)

#extrai os eventos já marcados nos arquivos
id_eventos = dict(T1=1, T2=2)

eventos, _ = mne.events_from_annotations(raw)

#divide o sinal em partes menores (epochs)
epochs = mne.Epochs(
    raw,
    eventos,
    event_id=id_eventos,
    tmin=0.5,
    tmax=2.5,
    baseline=None,
    preload=True)
print(epochs.get_data().shape)

#definindo canais importantes
canais_importantes = ['C3..', 'C4..', 'Cz..']
epochs = epochs.pick(canais_importantes)

#transforma dados para ML
dados = epochs.get_data()
labels = epochs.events[:,-1]

X = []
Y = labels
for sample in dados:
    features = []
    for canal in sample:
        var = np.var(canal)
        log_var = np.log(var)
        features.append(log_var)
    X.append(features)
X = np.array(X)

#normaliza os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

#balanceando as classes
# juntar X e y
XY = np.hstack((X, Y.reshape(-1, 1)))

classe1 = XY[XY[:, -1] == 1]
classe2 = XY[XY[:, -1] == 2]

# reduzir classe maior
classe1_down = resample(classe1,
                        replace=False,
                        n_samples=len(classe2),
                        random_state=42)

Xy_bal = np.vstack((classe1_down, classe2))

# separar novamente
X = Xy_bal[:, :-1]
Y = Xy_bal[:, -1]

#treina o modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)   #separa o treino do teste
modelo = SVC(kernel='rbf', class_weight='balanced')  #cria o modelo
modelo.fit(X_train, Y_train)        #treina  modelo

#avaliar
Y_pred = modelo.predict(X_test)
print("Acurácia:", accuracy_score(Y_test, Y_pred)) #acertos/total
print(classification_report(Y_test, Y_pred))

#visualizacao
plt.plot(Y_test[:50], label="Real")
plt.plot(Y_pred[:50], label="Previsto")
plt.legend()
plt.show

labels = epochs.events[:, -1]

from collections import Counter
print(Counter(labels))