# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 23:53:51 2020

@author: lsilvape
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('teste.csv')

previsores = df.iloc[:,0:4].values
classe = df.iloc[:,4].values

labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


classificador = GaussianNB()
#gerar a tabela de probabilidade
classificador.fit(previsores, classe)

print(classificador.classes_)
print(classificador.class_count_)

resultado = classificador.predict([[1,0,0,2]])

