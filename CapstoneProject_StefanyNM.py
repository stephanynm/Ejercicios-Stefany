#!/usr/bin/env python
# coding: utf-8

# In[1]:


##import
import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt

##estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model

##Model metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

##cross validation
from sklearn.model_selection import train_test_split


# In[2]:


nota = pd.read_csv(r'C:\Users\Eric\Documents\Data Analytics\Parte 5\5. Proyecto final\student-mat.csv')


# PREPARACIÓN DE LA INFORMACIÓN

# Los datos se obtuvieron en una encuesta de estudiantes de matemáticas en la escuela secundaria. Contiene mucha información social, de género y de estudio interesante sobre los estudiantes. 

# In[3]:


nota.head()


# In[4]:


nota.describe()


# In[5]:


nota.info()


# In[5]:


## Preprocesamiento
nota.dropna() ##quitar filas con variables nulas


# In[6]:


##Cambiar variables a categóricas
nota = nota.astype({"school":'category',"sex":'category', "address":'category', "famsize":'category', "Pstatus":'category', "Medu":'category', "Fedu":'category', "Mjob":'category', "Fjob":'category', "reason":'category', "guardian":'category', "failures":'category',"schoolsup":'category', "famsup":'category', "paid":'category', "activities":'category',"nursery":'category',"higher":'category',"internet":'category',"romantic":'category',"famrel":'category', "freetime":'category',"goout":'category',"Dalc":'category',"Walc":'category',"health":'category'})


# In[7]:


##Nuevas variables categóricas para las notas
##si es menor a 12 reprueba, 1:reprueba y 2:aprueba
def pass_faild(N1):
    G1, school = N1
    
    if G1 < 12:
        return '1'
    else:
        return '2'
nota['N1'] = nota[['G1', 'school']].apply(pass_faild, axis=1)
nota[:3]


# In[8]:


def pass_faild(N2):
    G2, school = N2
    
    if G2 < 12:
        return '1'
    else:
        return '2'
nota['N2'] = nota[['G2', 'school']].apply(pass_faild, axis=1)
nota[:3]


# In[9]:


def pass_faild(N3):
    G3, school = N3
    
    if G3 < 12:
        return '1'
    else:
        return '2'
nota['N3'] = nota[['G3', 'school']].apply(pass_faild, axis=1)
nota[:3]


# In[232]:


nota2 = nota.loc[:, ['school', 'sex', 'age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','N1','N2','N3']]


# In[233]:


nota2[:5]


# In[18]:


nota2.N1= nota2.N1.astype("category")
nota2.N2= nota2.N2.astype("category")
nota2.N3= nota2.N3.astype("category")
nota2.dtypes ##tipos de valores


# ANÁLISIS DESCRIPTIVO DE LA INFORMACIÓN

# In[19]:


nota2.describe()


# In[38]:


pd.unique(nota2['studytime'])


# In[39]:


nota2.studytime= nota2.studytime.astype("category")


# In[80]:


import seaborn as sns 


# In[158]:


fg = sns.factorplot('N3', data=nota2, kind='count', aspect=1.5)
fg.set_xlabels('resultado')


# 59% de reprobados

# In[122]:


sns.factorplot('studytime', data=nota2, kind='count', hue='N3', order=[1,2,3,4], 
               hue_order=['1','2'], aspect=2)


# In[157]:


# resultado de nota final por tiempo de estudio
nota2.groupby(['studytime', 'N3'])['studytime'].count()


# Para los que estudian 2 hrs hay mayor proporción de reprobados 64%, para el resto de horas de estudio se mantiene cercano al 50 y 50%
# 

# In[98]:


sns.factorplot('Dalc', data=nota2, kind='count', hue='N3', order=[1,2,3,4,5], 
               hue_order=['1','2'], aspect=2)


# In[162]:


# resultado de nota final por Workday alcohol consumption
nota2.groupby(['Dalc', 'N3'])['Dalc'].count()


# No hay diferencia significativa en la proporción de aprobados y reprobados para los diferentes niveles de consumo de alcohol en días laborales, en general 59% reprobados vs 41% aprobados, se eleva un poco el % de reprobados en el segundo nivel a un 68%

# In[115]:


sns.factorplot('romantic', data=nota2, kind='count', hue='N3', order=["yes","no"], 
               hue_order=['1','2'], aspect=2)


# In[163]:


# resultado de nota final por romantic relationship
nota2.groupby(['romantic', 'N3'])['romantic'].count()


# No hay una diferencia significativa con respecto a la proporción de reprobados al tener o no una relación romántica, general 59% de reprobados

# In[116]:


sns.factorplot('Walc', data=nota2, kind='count', hue='N3', order=[1,2,3,4,5], 
               hue_order=['1','2'], aspect=2)


# In[165]:


# resultado de nota final por Weekend alcohol consumption
nota2.groupby(['Walc', 'N3'])['Walc'].count()


# Hay un leve aumento en la proporción de reprobados para los jóvenes que tienen un mayor nivel de consumo de alcohol los fines de semana, para el nivel 4 un 71% de reprobados y para el nivel 5 un 65%, vs el promedio de 59%.
# La población está mas distribuido entre todos los niveles con respecto a la variable de consumo de alcohol en workday.

# In[78]:


fg = sns.factorplot('Fjob', data=nota2, kind='count', aspect=1.5)
fg.set_xlabels('resultado')


# In[166]:


fg = sns.factorplot('Mjob', data=nota2, kind='count', aspect=1.5)
fg.set_xlabels('resultado')


# In[79]:


fg = sns.factorplot('age', data=nota2, kind='count', aspect=1.5)
fg.set_xlabels('resultado')


# In[169]:


sns.factorplot('Mjob', data=nota2, kind='count', hue='N3', order=["at_home","health","other","services","teacher"], 
               hue_order=['1','2'], aspect=2)


# In[171]:


# resultado de nota final por profesión de la madre
nota2.groupby(['Mjob', 'N3'])['Mjob'].count()


# No hay diferencia en la proporción de reprobados, con respecto al promedio, si la profesión de la madre es teacher.

# In[170]:


sns.factorplot('Fjob', data=nota2, kind='count', hue='N3', order=["at_home","health","other","services","teacher"], 
               hue_order=['1','2'], aspect=2)


# In[172]:


# resultado de nota final por profesión del padre
nota2.groupby(['Fjob', 'N3'])['Fjob'].count()


# Cuando la profesión del padre es tecaher aumenta la proporción de aprobados a un 59% comparado con un 41% del promedio, es un aumento significativo, pero la muestra es pequeña.

# In[173]:


sns.factorplot('sex', data=nota2, kind='count', hue='N3', order=["F","M"], 
               hue_order=['1','2'], aspect=2)


# In[174]:


# resultado de nota final por sexo
nota2.groupby(['sex', 'N3'])['sex'].count()


# Hay una mayor proporción de mujeres que reprueban el curso, un 64% vs los hombres que reprueban en un 51% de los casos.

# In[175]:


sns.factorplot('guardian', data=nota2, kind='count', hue='N3', order=["father","mother","other"], 
               hue_order=['1','2'], aspect=2)


# In[177]:


# resultado de nota final por guardian
nota2.groupby(['guardian', 'N3'])['guardian'].count()


# Hay mayor proporción de reprobados cuando la persona a cargo no es el padre o la madre, un 72%

# In[178]:


sns.factorplot('failures', data=nota2, kind='count', hue='N3', order=[0,1,2,3], 
               hue_order=['1','2'], aspect=2)


# In[179]:


# resultado de nota final por ausencias
nota2.groupby(['failures', 'N3'])['failures'].count()


# El % de reprobados aumenta gradualmente conforme a aumentan las ausencias a clases.

# In[188]:


fig = sns.FacetGrid(nota2, hue='sex', aspect=4)
fig.map(sns.kdeplot, 'age', shade=True)
oldest = nota2['age'].max()
fig.set(xlim=(12,oldest))
fig.set(title='Distribution of age Grouped by sex')
fig.add_legend()


# In[189]:


fig = sns.FacetGrid(nota2, hue='studytime', aspect=4)
fig.map(sns.kdeplot, 'age', shade=True)
oldest = nota2['age'].max()
fig.set(xlim=(12,oldest))
fig.set(title='Distribution of age Grouped by studytime')
fig.add_legend()


# Los más jóvenes dedican más horas a estudiar

# In[243]:


sns.factorplot('schoolsup', data=nota2, kind='count', hue='N3', order=["yes","no"], 
               hue_order=['1','2'], aspect=2)


# In[245]:


# resultado de nota final por pago clases extra
nota2.groupby(['schoolsup', 'N3'])['schoolsup'].count()


# Las personas que pagaron clases extra tuvieron una mayor proporción de fracaso

# DUMMY'S

# In[225]:


## Dummy's 
nota['school2']= nota.school.map({'GP':0,'MS':1})
nota['sex2']= nota.sex.map({'F':0,'M':1})
nota['address2']= nota.address.map({'R':0,'U':1})                                   
nota['famsize2']= nota.famsize.map({'GT3':0,'LE3':1})                                   
nota['Pstatus2']= nota.Pstatus.map({'A':0,'T':1})                                   
nota['Mjob2']= nota.Mjob.map({'at_home':0,'health':1,'other':2,'services':3,'teacher':4})                                   
nota['Fjob2']= nota.Fjob.map({'at_home':0,'health':1,'other':2,'services':3,'teacher':4}) 
nota['reason2']= nota.reason.map({'course':0,'other':1,'home':1,'reputation':1})
nota['guardian2']= nota.guardian.map({'mother':0,'father':1,'other':2})                                   
nota['schoolsup2']= nota.schoolsup.map({'yes':0,'no':1})                                   
nota['famsup2']= nota.famsup.map({'yes':0,'no':1})                                   
nota['paid2']= nota.paid.map({'yes':0,'no':1})                                   
nota['activities2']= nota.activities.map({'yes':0,'no':1})
nota['nursery2']= nota.nursery.map({'yes':0,'no':1})                                   
nota['higher2']= nota.higher.map({'yes':0,'no':1})                                   
nota['internet2']= nota.internet.map({'yes':0,'no':1})                                   
nota['romantic2']= nota.romantic.map({'yes':0,'no':1}) 


# In[226]:


nota3 = nota.loc[:, ['school2', 'sex2', 'age','address2','famsize2','Pstatus2','Medu','Fedu','Mjob2','Fjob2','reason2','guardian2','traveltime','studytime','failures','schoolsup2','famsup2','paid2','activities2','nursery2','higher2','internet2','romantic2','famrel','freetime','goout','Dalc','Walc','health','absences','N1','N2','N3']]


# In[227]:


nota3[:5]


# SELECCIÓN DE CARACTERÍSTICAS

# In[228]:


import sklearn as skl
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


# In[230]:


N3 = nota3['N3']
# Aplicando el algoritmo univariante de prueba F.
k = 20  # número de atributos a seleccionar
entrenar = nota3.drop(['N3'], axis=1)
columnas = list(entrenar.columns.values)
seleccionadas = SelectKBest(f_classif, k=k).fit(entrenar, N3)
atrib = seleccionadas.get_support()
atributos = [columnas[i] for i in list(atrib.nonzero()[0])]
atributos


# In[240]:


from sklearn.ensemble import ExtraTreesClassifier
# Algoritmo de Eliminación Recursiva de atributos con ExtraTrees
modelo = ExtraTreesClassifier()
era = RFE(modelo, 22)  # número de atributos a seleccionar
era = era.fit(entrenar, N3)
# imprimir resultados
atrib = era.support_
atributos = [columnas[i] for i in list(atrib.nonzero()[0])]
atributos


# In[241]:


# Importancia de atributos.
modelo.fit(entrenar, N3)
modelo.feature_importances_[:22]


# In[246]:


features = nota3.loc[:, ['sex2', 'age', 'Medu','Fedu','Mjob2','Fjob2','reason2','traveltime','studytime','failures','schoolsup2','famsup2','famrel','freetime','goout','Dalc','Walc','health','absences','N1','N2']]


# In[248]:


#dependent variable
depVar = nota3['N3']


# MODELOS

# In[293]:


# Modelos 
modelSVC = SVC(kernel='rbf', C=1) 
modelRF = RandomForestClassifier(n_estimators=2, criterion='gini',) 
modelKN = KNeighborsClassifier(n_neighbors = 2, metric = 'euclidean', p = 2, weights = 'uniform')


# In[251]:


x_train, x_test, y_train, y_test = train_test_split(features, depVar, test_size = 0.3,random_state = 0)


# In[262]:


modelSVCfit= modelSVC.fit(x_train, y_train)


# In[263]:


modelSVCfit


# In[265]:


modelRFfit= modelRF.fit(x_train, y_train)
modelRFfit


# In[294]:


modelKNfit= modelKN.fit(x_train, y_train)
modelKNfit


# In[258]:


##Elegir el mejor valor de k
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])


# In[264]:


print(cross_val_score(modelSVCfit, x_train, y_train))
modelSVCfit.score (x_train, y_train)


# In[267]:


print(cross_val_score(modelRFfit, x_train, y_train))
modelRFfit.score (x_train, y_train)


# In[295]:


print(cross_val_score(modelKNfit, x_train, y_train))
modelKNfit.score(x_train, y_train)


# PREDICCIONES

# In[270]:


prediccionSVC = modelSVC.predict(x_test)
accuracy_score(y_test, prediccionSVC)


# In[271]:


prediccionRF = modelRF.predict(x_test)
accuracy_score(y_test, prediccionRF)


# In[272]:


prediccionKN = modelKN.predict(x_test)
accuracy_score(y_test, prediccionKN)


# In[273]:


prediccionKN


# In[276]:


confusion_matrix(y_test, prediccionSVC)


# In[277]:


confusion_matrix(y_test, prediccionRF)


# In[274]:


confusion_matrix(y_test, prediccionKN)


# In[278]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediccionSVC))


# In[279]:


print(classification_report(y_test, prediccionRF))


# In[275]:


print(classification_report(y_test, prediccionKN))


# SELECCIONAR MODELO KN
