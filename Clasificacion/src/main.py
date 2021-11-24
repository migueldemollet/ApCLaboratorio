from typing import Any
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score, f1_score, accuracy_score, roc_curve, classification_report, auc, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import copy
import warnings
warnings.filterwarnings('ignore')

def load_db(dir_db: str) -> pd.DataFrame:
    """
    Esta función se encarga de leer los ficheros csv y transformarlo en variables de tipo pd.DataFrame gracias a la ayuda 
    de la librería de pandas
    
    Parámetros
    -----------
    Dir_db: str
    String que indica el directorio de la base de datos para el entreno del modelo

    Returns
    -----------
    pd.DataFrame: dataframe del directorio de entreno
    """
    #Obligamos que muestre todas las columnas
    pd.set_option('display.max_columns', None)
    # Visualizamos solo 3 deciamles
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    dataset = pd.read_csv(dir_db, header=0, delimiter=',')
    
    return dataset

def split_data(X: np.ndarray, Y: np.ndarray, train_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Esta función se encarga de dividir los datos de entrenamiento y prueba de la base de datos
    
    Parámetros
    -----------
    X: np.ndarray
    Matriz de datos de entrada

    Y: np.ndarray
    Matriz de datos de salida

    train_size: float
    Porcentaje de datos de entrenamiento

    Returns
    -----------
    pd.DataFrame: Dataset de entrenamiento
    pd.DataFrame: Dataset de prueba
    pd.DataFrame: Dataset de salida de entrenamiento
    pd.DataFrame: Dataset de salida de prueba
    """
    return train_test_split(X, Y, train_size=train_size)

def logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Esta función es la encargada de entrenar el modelo logistico y evaluarlo con los datos de prueba

    Parámetros
    -----------
    X_train: pd.DataFrame
    Dataset de entrenamiento

    X_test: pd.DataFrame
    Dataset de prueba

    y_train: pd.DataFrame
    Dataset de salida de entrenamiento

    y_test: pd.DataFrame
    Dataset de salida de prueba

    Returns
    -----------
    Ninguno
    """
    regresion_logistica= LogisticRegression()
    tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'penalty':['l1','l2']}
    regresion_logistica= GridSearchCV(regresion_logistica, tuned_parameters,cv=10)
    regresion_logistica.fit(X_train, y_train)

    y_prob = regresion_logistica.predict_proba(X_test)[:,1]  
    y_pred = np.where(y_prob > 0.5, 1, 0) 
    print(f'Score: ', regresion_logistica.score(X_test, y_pred))

    #generating roc_curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8,8))
    plt.title('Curva de ROC Regression lineal')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def svm_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Esta función es la encargada de entrenar el modelo SVM y evaluarlo con los datos de prueba

    Parámetros
    -----------
    X_train: pd.DataFrame
    Dataset de entrenamiento

    X_test: pd.DataFrame
    Dataset de prueba

    y_train: pd.DataFrame
    Dataset de salida de entrenamiento

    y_test: pd.DataFrame
    Dataset de salida de prueba

    Returns
    -----------
    Ninguno
    """
    tuned_parameters = {
        'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
        'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf']
    }
    svm_model=SVC()
    model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=30)
    model_svm.fit(X_train, y_train)
    y_pred= model_svm.predict(X_test)
    print(f'Score: ', model_svm.score(X_test, y_pred))

    #generating roc_curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8,8))
    plt.title('Curva de ROC SVM')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Esta función es la encargada de entrenar el modelo Random Forest y evaluarlo con los datos de prueba

    Parámetros
    -----------
    X_train: pd.DataFrame
    Dataset de entrenamiento

    X_test: pd.DataFrame
    Dataset de prueba

    y_train: pd.DataFrame
    Dataset de salida de entrenamiento

    y_test: pd.DataFrame
    Dataset de salida de prueba

    Returns
    -----------
    Ninguno
    """
    clf = RandomForestClassifier(n_estimators=300,min_samples_leaf=0.15)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Score: ', clf.score(X_test, y_pred))

    #generating roc_curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(8,8))
    plt.title('Curva de ROC Random Forest')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def compare_models_by_precision(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    """
    Esta función es la encargada de comparar los modelos de entrenamiento con la metrica de precision
    
    Parámetros
    -----------
    X_train: pd.DataFrame
    Dataset de entrenamiento

    y_train: pd.DataFrame
    Dataset de salida de entrenamiento

    Returns
    -----------
    Ninguno
    """
    modelos_test = [SVC(kernel='linear',C =100),
          RandomForestClassifier(n_estimators=300,random_state=40),
          KNeighborsClassifier(), 
          LogisticRegression(max_iter=1000,solver='lbfgs'),
          DecisionTreeClassifier(),
          GaussianNB(),]

    modelos = ['SVC', 'Random Forest','KNN',  'Regresion Logistica','DecisionTreeClassifier','GaussianNB']
    scores_unscaled = []
    for index,model in enumerate(modelos_test):
        try:
            model.fit(X_train,y_train)
            print("Precision =",modelos[index] ,":",round(model.score(X_train,y_train)*100,2),"%")
            scores_unscaled.append(round(model.score(X_train,y_train)*100,2))
        except:
            print("Skipped",modelos[index])
    
    #mostrando grafica de los modelos
    plt.plot(range(len(modelos)),scores_unscaled, '-o')
    plt.xticks(range(0,6,1),labels = modelos, rotation = 90)
    plt.grid(visible=True)
    plt.show()

def compare_models(dataset: pd.DataFrame) -> None:
    """
    Esta función es la encargada de comparar el modelo de regression lineal junto al de svm

    Parámetros
    -----------
    dataset: pd.DataFrame
    Dataset de entrenamiento

    Returns
    -----------
    Ninguno
    """
    X =  dataset[['thalachh','cp']].values
    y = dataset['output']
        
    fig, sub = plt.subplots(1, 2, figsize=(16,6))
    sub[0].scatter(X[:,0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    sub[1].scatter(X[:,1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    particions = [0.5, 0.7, 0.8]

    for part in particions:
        x_t, x_v, y_t, y_v = split_data(X, y, part)
        
        #Creacion del regresor logistico 
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
        logireg.fit(x_t, y_t)
        print ("Correct classification Logistic ", part, "% of the data: ", logireg.score(x_v, y_v))
        
        #Utilizacion de máquinas de vectores de soporte
        svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)
        svc.fit(x_t, y_t)
        print ("Correct classification SVM      ", part, "% of the data: ", svc.score(x_v, y_v))

    plt.show()

def columns_with_na(dframe: pd.DataFrame) -> int:
    """
    Función encargada de buscar los índices de las columnas que tengan algún valor nulo dentro del dataframe
    
    Parámetros
    -----------
    dframe: pd.DataFrame
    dataframe al que buscaremos los valores nulos
    
    Returns
    -----------
    Int: índices de las columnas nulas
    """
    temp = dframe.isna().sum()
    temp = temp[temp >0]

    return temp.index

def prepare_data(dataset: pd.DataFrame) -> None:
    """
    Esta función es la encargada de preparar el dataset para su análisis, así como limpiar los valores nulos 
    usando la ayuda de la función columns_with_na y mostrando de forma gráfica una vez limpiada
    
    Parámetros
    -----------
    dataset: pd.DataFrame 
    dataset “sucio”
    
    Returns
    -----------
    Ninguno
    """
    if (dataset.isnull().sum().any() > 0):
        columns_to_drop = list(columns_with_na(dataset))
        dataset.drop(columns_to_drop, axis=1, inplace = True)
        sns.heatmap(dataset.isnull(), cbar=False)

def dataset_statistics(dataset: pd.DataFrame) -> None:
    """
    Esta función es vital para nuestro análisis de los datos ya que es la encargada de mostrar datos estadísticos como por ejemplo características del dataset, las 5 primeras filas de este, etc. Y los pintamos por terminal
    
    Parámetros
    -----------
    dataset: pd.DataFrame
    dataset que queremos explorer
    
    Returns
    -----------
    Ninguno
    """
    print("=========Informacion=========")
    print(dataset.info(), end="\n")

    print("=========Los 5 primeros elementos=========")
    print(dataset.head(), end="\n")

    print("=========Descripcion=========")
    print(dataset.describe(), end="\n")

    print("=========Dimensiones=========")
    print(dataset.shape, end="\n")

def dataset_graphics(dataset: pd.DataFrame) -> None:
    """
    Función encargada de mostrar gráficamente los datos del dataset
    
    Parámetros
    -----------
    dataset: pd.DataFrame
    dataset que queremos explorer
    
    Returns
    -----------
    Ninguno
    """
    print("=========Frecuencia de padecer del corazon según el edad=========")
    fig_dims = (15, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(dataset['age'],dataset['output'],ax=ax)
    plt.show()

    print("=========Frecuencia cardíaca máxima alcanzada segun la edad=========")
    plt.subplots(figsize=(10, 5))
    plt.scatter(x=dataset.age[dataset.output==1], y=dataset.thalachh[(dataset.output==1)], c="red")
    plt.scatter(x=dataset.age[dataset.output==0], y=dataset.thalachh[(dataset.output==0)], c="green")
    plt.legend(["Padece de enfermedad", "No padece de enfermedad"])
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia cardíaca máxima alcanzada")
    plt.title("Frecuencia cardíaca máxima alcanzada segun la edad")
    plt.show()

    print("=========Frecuencia cardíaca máxima alcanzada segun el sexo=========")
    pd.crosstab(dataset.sex,dataset.output).plot(kind="bar",figsize=(10,6),color=['#67a1cf','#e3a6c3' ])
    plt.title('Frecuencia de enfermedad cardíaca segun el sexo')
    plt.xlabel('Sexo [0:Femenino, 1: Maculino]')
    plt.legend(["No padece de enfermedad", "Padece de enfermedad"])
    plt.ylabel('Frecuecia')
    plt.show()

    print("=========Frecuencia de padecer del corazon según el tipo de dolor en el pecho=========")
    pd.crosstab(dataset.cp,dataset.output).plot(kind="bar",figsize=(15,6),color=['#ca6457','#6f2923' ])
    plt.title('Frecuencia de padecer del corazon según el tipo de dolor en el pecho')
    plt.legend(["No padece de enfermedad", "Padece de enfermedad"])
    plt.xlabel('Tipo de dolor en el pecho [0:typical angina, 1:atypical angina, 2:non-anginal pain, 3:asymptomatic]')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequencia de padecer la enfermedad o no')
    plt.show()

    print("=========Frecuencia de padecer del corazon según el nivel de azucar en sangre=========")
    pd.crosstab(dataset.fbs,dataset.output).plot(kind="bar",figsize=(15,6),color=['#6ad0a6','#226d5c' ])
    plt.title('Frecuencia de padecer del corazon según el nivel de azucar en sangre')
    plt.legend(["No padece de enfermedad", "Padece de enfermedad"])
    plt.xlabel('Nivel de azucar en sangre [0: > 120 mg/dl]')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequencia de padecer la enfermedad o no')
    plt.show()
    
    print("=========Frecuencia de padecer del corazon según resultados electrocardiográficos en reposo=========")
    pd.crosstab(dataset.restecg,dataset.output).plot(kind="bar",figsize=(15,6),color=['#7a5dcc','#412168' ])
    plt.title('Frecuencia de padecer del corazon según resultados electrocardiográficos en reposo')
    plt.legend(["No padece de enfermedad", "Padece de enfermedad"])
    plt.xlabel('Tipo de resultados')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequencia de padecer la enfermedad o no')
    plt.show()

    print("=========Frecuencia de padecer del corazon según Talisemia=========")
    pd.crosstab(dataset.thall,dataset.output).plot(kind="bar",figsize=(15,6),color=['#b8df9b','#4d5e1e' ])
    plt.title('Frecuencia de padecer del corazon según tipo de Talasemia')
    plt.legend(["No padece de enfermedad", "Padece de enfermedad"])
    plt.xlabel('Tipo de Talasemia')
    plt.xticks(rotation = 0)
    plt.ylabel('Frequencia de padecer la enfermedad o no')
    plt.show()

    print("=========Correlaion de los atributos=========")
    correlacion = dataset.iloc[:].corr()
    fig , ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    sns.heatmap(correlacion,annot=True,cmap="coolwarm")
    plt.show()

def dataset_atipic_values(dataset: pd.DataFrame) -> None:
    """
    Función que muestra la distribucion de los valores atípicos en el dataset

    Parameters
    -----------
    dataset: pd.DataFrame
    Dataset con los datos
    
    Returns
    --------
    Ninguno
    """
    fig, ax = plt.subplots(ncols = 7, nrows = 2, figsize = (20, 10))
    index = 0
    ax = ax.flatten()

    for col, value in dataset.items():
        sns.boxplot(y=col, data=dataset, ax=ax[index])
        index += 1
    plt.tight_layout(pad = 0.5, w_pad=0.7, h_pad=5.0)
    plt.title("Distribución de los valores atípicos")
    plt.show()

def normalization(dataset: pd.DataFrame) -> None:
    """
    funcion que nonraliza los datos

    Parameters
    -----------
    dataset: pd.DataFrame
    Dataset con los datos

    Returns
    --------
    Ninguno
    """
    dataset.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def add_category_values(dataset: pd.DataFrame) -> None:
    """
    Función que agrega al dataset la clase que pertenece cada valore categorico

    Parameters
    -----------
    dataset: pd.DataFrame
    Dataset con los datos

    Returns
    --------
    Ninguno
    """
    dataset.drop(['sex', 'chol', 'restecg', 'fbs'], axis=1, inplace=True)
    #CP: type of chest pain 
    dataset.loc[dataset['cp'] == 0, 'cp'] = 'typical angina'
    dataset.loc[dataset['cp'] == 1, 'cp'] = 'atypical angina'
    dataset.loc[dataset['cp'] == 2, 'cp'] = 'non-anginal pain'
    dataset.loc[dataset['cp'] == 3, 'cp'] = 'asymptomatic'
    # enxercise induced angina
    dataset.loc[dataset['exng'] == 1, 'exng'] = 'yes'
    dataset.loc[dataset['exng'] == 0, 'exng'] = 'no'
    #slope of peak exercise ST segment
    dataset.loc[dataset['slp'] == 0, 'slp'] = 'upsloping'
    dataset.loc[dataset['slp'] == 1, 'slp'] = 'flat'
    dataset.loc[dataset['slp'] == 2, 'slp'] = 'downsloping'
    #thalassemia blood disorder
    dataset.loc[dataset['thall'] == 0, 'thall'] = 'None'
    dataset.loc[dataset['thall'] == 1, 'thall'] = 'fixed defect'
    dataset.loc[dataset['thall'] == 2, 'thall'] = 'normal blood flow'
    dataset.loc[dataset['thall'] == 3, 'thall'] = 'reversible defect'

def one_hot_encoding(dataset: pd.DataFrame) -> None:
    """
    Función que codifica los valores categóricos con one hot encoding

    Parameters
    -----------
    dataset: pd.DataFrame
    Dataset con los datos

    Returns
    --------
    Ninguno
    """
    X=dataset.iloc[:, :-1]
    y=dataset.iloc[:, -1]
    x_train, x_test, y_train, y_test = split_data(X, y, 0.9)

    encoding_cols = ['cp', 'exng', 'slp', 'thall']
    oh_encode = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

    x_train_encoded = pd.DataFrame(oh_encode.fit_transform(x_train[encoding_cols]), index = x_train.index)
    x_test_encoded = pd.DataFrame(oh_encode.transform(x_test[encoding_cols]), index = x_test.index)

    x_train_num = x_train.drop(['cp', 'exng', 'slp', 'thall'], axis=1)
    x_test_num = x_test.drop(['cp', 'exng', 'slp', 'thall'], axis=1)

    x_train_en = pd.concat([x_train_num, x_train_encoded], axis=1)
    x_test_en = pd.concat([x_test_num, x_test_encoded], axis=1)
    print(x_train_en.head())

def train_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, model_name: str) -> tuple[Any, dict[str, Any], float]:
    """
    Función que entrena el modelo, muestra metricas de este y su matrix de confusion

    Parameters
    -----------
    model: Any
    Modelo a entrenar

    X_train: pd.DataFrame
    Dataset con los datos de entrenamiento

    X_test: pd.DataFrame
    Dataset con los datos de prueba

    y_train: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    y_test: pd.DataFrame
    Dataset con los datos de salida de prueba

    model_name: str
    Nombre del modelo

    Returns
    --------
    Any: modelo entrenado
    dict[str, Any]: nombre del modelo y su accuracy
    float: accuracy del modelo
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    valor={"Modelo":model_name,"Precision":accuracy_score(y_test, y_pred)}
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    cm_rl=confusion_matrix(y_test, y_pred)
    sns.heatmap(cm_rl/np.sum(cm_rl),annot = True, fmt=  '0.2%',cmap ='Purples')
    plt.show()

    return model, valor, y_pred

def statics_model_regression_logistic(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Función que muestra las estadisticas de un modelo de regresión logística

    Parameters
    -----------
    X_train: pd.DataFrame
    Dataset con los datos de entrenamiento

    X_test: pd.DataFrame
    Dataset con los datos de prueba

    y_train: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    y_test: pd.DataFrame
    Dataset con los datos de salida de prueba

    Returns
    --------
    Ninguno
    """
    modelos_rl = pd.DataFrame(columns=["Modelo","Precision","C","fit_intercept","penalty","tol"])
    c_list=[0.001,0.01,0.1,1.0,2.0]
    f_inter=[False,True]
    for i in c_list:
        for j in f_inter:
            #Modelo
            modelo=LogisticRegression(C=i,fit_intercept=j, penalty='l2',tol=0.001)
            #Entrenamiento del modelo
            modelo.fit(X_train,y_train)
            #Prediccion en base al training test
            prediccion=modelo.predict(X_test)
            #Accuracy obtenido
            acc=accuracy_score(y_test,prediccion)
            valor2={"Modelo":"Regresion Logistica","Precision":acc,"C":i,"fit_intercept":j,"penalty":"l2","tol":0.001}
            modelos_rl =modelos_rl .append(valor2,ignore_index=True)
    
    print(modelos_rl.sort_values(by="Precision", ascending=False))

def statics_model_smv(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, kernel: str) -> None:
    """
    Función que muestra las estadisticas de un modelo SVM

    Parameters
    -----------
    X_train: pd.DataFrame
    Dataset con los datos de entrenamiento

    X_test: pd.DataFrame
    Dataset con los datos de prueba

    y_train: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    y_test: pd.DataFrame
    Dataset con los datos de salida de prueba

    kernel: str
    Tipo de kernel para el modelo SVM

    Returns
    --------
    Ninguno
    """
    modelos_svm =pd.DataFrame(columns=["Modelo","Precision","C","probability","gamma","tol"])
    c_list=[0.001,0.01,0.1,1.0,2.0]
    gammas=['scale','auto']
    for i in c_list:
        for j in gammas:
            #Modelo
            modelo_svm = SVC(C=i,kernel=kernel,tol=0.001,probability=True)
            #Entrenamiento del modelo
            modelo_svm.fit(X_train,y_train)
            #Prediccion en base al training test
            prediccion_svg=modelo_svm.predict(X_test)
            #Accuracy obtenido
            acc_svg=accuracy_score(y_test,prediccion_svg)
            valor2={"Modelo":"SVG_rbf","Precision":acc_svg,"C":i,"probability":'True','gamma':j,"tol":0.001}
            modelos_svm = modelos_svm.append(valor2,ignore_index=True)
    
    print(modelos_svm.sort_values(by="Precision", ascending=False))

def statics_model_random_forest(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Función que muestra las estadisticas de un modelo Random Forest

    Parameters
    -----------
    X_train: pd.DataFrame
    Dataset con los datos de entrenamiento

    X_test: pd.DataFrame
    Dataset con los datos de prueba

    y_train: pd.DataFrame
    Dataset con los datos de entrenamiento

    y_test: pd.DataFrame
    Dataset con los datos de prueba

    Returns
    --------
    Ninguno
    """
    modelos_rf=pd.DataFrame(columns=["Modelo","Precision","n_estimators","min_samples_leaf"])
    estimadores=[100,200,300,500]
    min_samples=[0.1,0.15,0.2,0.25,0.3]
    for i in estimadores:
        for j in min_samples:
            #Modelo
            modelo_rf = RandomForestClassifier(n_estimators=i,min_samples_leaf=j)
            #Entrenamiento del modelo
            modelo_rf.fit(X_train,y_train)
            #Prediccion en base al training test
            prediccion_rf=modelo_rf.predict(X_test)
            #Accuracy obtenido
            acc_rf=accuracy_score(y_test,prediccion_rf)
            valor={"Modelo":"Random Forest","Precision":acc_rf,"n_estimators":i,"min_samples_leaf":j}
            modelos_rf=modelos_rf.append(valor,ignore_index=True)
    
    print(modelos_rf.sort_values(by="Precision", ascending=False))

def statics_model_knn(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Función que muestra las estadisticas de un modelo KNN

    Parameters
    -----------
    X_train: pd.DataFrame
    Dataset con los datos de entrenamiento

    X_test: pd.DataFrame
    Dataset con los datos de prueba

    y_train: pd.DataFrame
    Dataset con los datos de salida entrenamiento

    y_test: pd.DataFrame
    Dataset con los datos de salida de prueba

    Returns
    --------
    Ninguno
    """
    modelos_knn=pd.DataFrame(columns=["Modelo","Precision","n"])
    n=[10,15,20,30,50]
    for i in n:
        #Modelo
        knn =  KNeighborsClassifier(n_neighbors=i)
        #Entrenamiento del modelo
        knn.fit(X_train, y_train)
        #Prediccion en base al training test
        prediccion_knn=knn.predict(X_test)
        #Accuracy obtenido
        acc_knn=accuracy_score(y_test,prediccion_knn)
        
        valor={"Modelo":"Knn","Precision":acc_knn,"n":i}
        modelos_knn=modelos_knn.append(valor,ignore_index=True)
    
    print(modelos_knn.sort_values(by="Precision", ascending=False))

def k_split(X: pd.DataFrame, y: pd.DataFrame) -> KFold:
    """
    Función que genera los k folds para el modelo de cross validation

    Parameters
    -----------
    X: pd.DataFrame
    Dataset con los datos de entrenamiento

    y: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    Returns
    --------
    KFold: Objeto que contiene los k folds
    """
    #Usemos cross_val_score para evaluar una puntuación mediante validación cruzada.
    kf =KFold(n_splits=5, shuffle=True, random_state=42)
    cnt = 1
    # split()  method generate indices to split data into training and test set.
    for train_index, test_index in kf.split(X, y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        cnt += 1

    return kf

def rmse(score: float) -> None:
    """
    Funcion que muestra el error cuadrático medio

    Parameters
    ----------
    score: float
    Score obtenido en el modelo
    
    Returns
    -------
    Ninguno
    """
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')

def model_estimator(X: pd.DataFrame, y: pd.DataFrame, kf: KFold, method_scoring) -> None:
    """
    Funcion que calcula el mejor ajuste para el random Forest y moostrando el error cuadrático medio por cada estimacion

    Parameters
    ----------
    X: pd.DataFrame
    Dataset con los datos de entrenamiento

    y: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    kf: KFold
    Objeto que contiene los k folds

    Returns
    -------
    Ninguno
    """
    #A partir de 250 es un buen estimador !
    estimators = [50, 100, 150, 200, 250]

    for count in estimators:
        score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= count, random_state= 42), X, y, cv= kf, scoring=method_scoring)
        print(f'For estimators: {count}')
        rmse(score.mean())

def leave_one_out(model_rf: LogisticRegression, X: pd.DataFrame, y: pd.DataFrame, method_score: str, splits: int) -> None:
    """
    Funcion que calcula el leave one out del modelos de regresion logistica

    Parameters
    ----------
    model_rf: LogisticRegression
    Modelo de regresión logística

    X: pd.DataFrame
    Dataset con los datos de entrenamiento

    y: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    method_score: str
    Método de evaluación

    splits: int
    Número de splits

    Returns
    -------
    Ninguno
    """
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    crossvalidation = KFold(n_splits=splits, random_state=None, shuffle=False)
    scores = cross_val_score(model_rf, X, y, scoring=method_score, cv=crossvalidation, n_jobs=1)
    print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))))

def evaluate_preds(y_true: pd.DataFrame, y_preds: pd.DataFrame) -> None:
    """
    Funcion que muestra las metricas de la prediccion del modelo (accuracy, precision, recall, f1)

    Parameters
    ----------
    y_true: pd.DataFrame
    Dataset con los datos de salida de prueba

    y_preds: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    Returns
    -------
    Ninguno
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    print(f"Acc : {round(accuracy, 2) * 100:.2f}%")
    print(f"Precision : {round(precision, 2):.2f}")
    print(f"recall : {round(recall, 2):.2f}")
    print(f"F1 score {round(f1, 2):.2f}")

def simple_roc_curve(y_test: pd.DataFrame, y_preds: pd.DataFrame , title: str) -> None:
    """
    Función que genera una curva ROC

    Parameters
    ----------
    y_test: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    y_preds: pd.DataFrame
    Dataset con los datos de predicción

    y_preds:

    title: str
    Título del gráfico

    Returns
    -------
    Ninguno
    """
    fpr, tpr, threshold = roc_curve(y_test, y_preds)
    plt.plot(fpr, tpr)
    plt.title(title)
    plt.xlabel('False positive Rate')
    plt.ylabel('True positive Rate')
    plt.grid(True)
    plt.show()

def simple_recall_curve(y_test: pd.DataFrame, y_preds: pd.DataFrame, title: str) -> None:
    """
    Función que genera una curva Recall

    Parameters
    ----------
    y_test: pd.DataFrame
    Dataset con los datos de salida de entrenamiento

    y_preds: pd.DataFrame
    Dataset con los datos de predicción

    title: str
    Título del gráfico

    Returns
    -------
    Ninguno
    """
    fpr, tpr, threshold = precision_recall_curve(y_test, y_preds)
    plt.plot(fpr, tpr)
    plt.title(title)
    plt.xlabel('False positive Rate')
    plt.ylabel('True positive Rate')
    plt.grid(True)
    plt.show()


def main():
    # Apartado B

    #Cargamos la base de datos
    dataset = load_db('db\\heart.csv')
    #entrenando modelo para comparativa
    print("=========Entrenando modelos para la comparativa=========")
    X = dataset.drop('output', axis=1)
    y=dataset['output']
    X_train, X_test, y_train, y_test = split_data(X, y, 0.5)

    print("=========Regression logistica=========")
    logistic_regression(X_train, X_test, y_train, y_test)

    print("=========SVM=========")
    svm_model(X_train, X_test, y_train, y_test)

    print("=========Random Forest=========")
    random_forest(X_train, X_test, y_train, y_test)

    #comparativa de modelos
    print("=========Comparativa de modelos por precision=========")
    compare_models_by_precision(X_train, y_train)

    print("=========Comparativa de modelos por score=========")
    compare_models(dataset)

    # Apartado A
    dataset_norm = copy.deepcopy(dataset)
    df_sense = copy.deepcopy(dataset)
    prepare_data(dataset)
    dataset_statistics(dataset)
    dataset_graphics(dataset)

    print("=========Distribución de los valores atípicos=========")
    dataset_atipic_values(dataset)

    print("=========Normalizaion de los datos=========")
    normalization(dataset_norm)
    dataset_statistics(dataset_norm)

    print("=========Distribución de los valores atípicos despues de normalizar=========")
    dataset_atipic_values(dataset_norm)

    print("=========Agregamos columnas categoricas a partir de valores categoricos=========")
    add_category_values(df_sense)

    print("=========Codificacion con One Hot Encoding=========")
    one_hot_encoding(df_sense)

    print("=========Seleccion del modelo=========")
    X=dataset_norm.drop('output', axis=1)
    y=dataset_norm['output']
    X_train, X_test, y_train, y_test = split_data(X, y, 0.8)
    modelos = pd.DataFrame(columns=["Modelo","Precision"])

    print("=========Regression logistica=========")
    model_lr = LogisticRegression(max_iter=200,random_state=0,n_jobs=30)
    model_lr, valor, y_pred_lr = train_model(model_lr, X_train, X_test, y_train, y_test, "Logistic Regression")
    modelos = modelos.append(valor, ignore_index=True)
    statics_model_regression_logistic(X_train,X_test,y_train, y_test)

    print("=========Regression SVC(rbf)=========")
    model_svc = SVC(kernel='rbf', random_state = 42)
    model_svc, valor, y_pred_svc = train_model(model_svc, X_train, X_test, y_train, y_test, "SVC(rbf)")
    modelos = modelos.append(valor, ignore_index=True)
    statics_model_smv(X_train,X_test,y_train, y_test, 'rbf')

    print("=========Regression SVC(linear)=========")
    model_svc2 = SVC(kernel='linear', random_state = 42)
    model_svc2, valor, y_pred_svc2 = train_model(model_svc2, X_train, X_test, y_train, y_test, "SVC(linear)")
    modelos = modelos.append(valor, ignore_index=True)
    statics_model_smv(X_train,X_test,y_train, y_test, 'linear')

    print("=========Regression SVC(poly)=========")
    model_svc3 = SVC(kernel='poly', random_state = 42)
    model_svc3, valor, y_pred_svc3 = train_model(model_svc3, X_train, X_test, y_train, y_test, "SVC(poly)")
    modelos = modelos.append(valor, ignore_index=True)
    statics_model_smv(X_train,X_test,y_train, y_test, 'poly')

    print("=========Random forest=========")
    model_rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=0.16, random_state=42)
    model_rf, valor, y_pred_rf = train_model(model_rf, X_train, X_test, y_train, y_test, "Random forest")
    modelos = modelos.append(valor, ignore_index=True)
    statics_model_random_forest(X_train,X_test,y_train, y_test)

    print("=========KNN=========")
    model_knn = KNeighborsClassifier(n_neighbors=17)
    model_knn, valor, y_pred_knn = train_model(model_knn, X_train, X_test, y_train, y_test, "KNN")
    modelos = modelos.append(valor, ignore_index=True)
    statics_model_knn(X_train,X_test,y_train, y_test)

    print("=========Mejores modelos by Precision=========")
    print(modelos.sort_values(by="Precision", ascending=False))

    # Crossvalidation
    print("=========Crossvalidacion con rmse=========")
    #Asignacion de variable independientes
    X=dataset_norm.drop('output', axis=1)
    y=dataset_norm['output']

    print("=========Dividimos los datos de entreno=========")
    kf = k_split(X, y)

    print("=========Evaluamos del Random forest=========")
    score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")
    print(f'Scores para cada fold son: {score}')
    rmse(score.mean())
    #ajuste del modelo
    model_estimator(X, y, kf, 'neg_mean_squared_error')

    print("=========Evaluamos de la regresion logistica=========")
    score = cross_val_score(LogisticRegression(), X, y, cv= kf, scoring="neg_mean_squared_error")
    print(f'Scores para cada fold son: {score}')
    rmse(score.mean())

    print("=========Crossvalidaion en base al acuracy=========")
    print("=========Evaluamos del Random forest=========")
    score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores para cada fold son: {score}')
    print(f'Average score: {"{:.2f}".format(score.mean())}')
    model_estimator(X, y, kf, 'accuracy')

    print("=========Evaluacion regression logistia=========")
    score = cross_val_score(LogisticRegression(), X, y, cv= kf, scoring="accuracy")
    print(f'Scores para cada fold son: {score}')
    print(f'Average score: {"{:.2f}".format(score.mean())}')

    print("=========Implementacion del LOO=========")
    leave_one_out(model_rf, X, y, 'neg_mean_squared_error', 5)
    leave_one_out(model_rf, X, y, 'accuracy', 10)

    #analisis de metricas
    print("=========Analisis de metricas random forest=========")
    evaluate_preds(y_test, y_pred_rf)

    print("=========Analisis de metricas con menos precision (regresion logistica)=========")
    evaluate_preds(y_test, y_pred_lr)

    #roc curve
    print("=========ROC curve=========")
    simple_roc_curve(y_test, y_pred_lr, 'ROC curve for Heart disease Logistic R classifier')
    simple_roc_curve(y_test, y_pred_rf, 'ROC curve for Heart disease Random Forest classifier')

    print("=========Recall curve=========")
    simple_recall_curve(y_test, y_pred_rf, 'Recall curve for Heart disease Random Forest classifier')

    print("=========Classificacion report random forest=========")
    target_names = ['Padece de enfermedad', 'No padece de enfermedad']
    print(classification_report(y_test, y_pred_rf, target_names=target_names))

    print("=========Classificacion report regression logistica=========")
    print(classification_report(y_test, y_pred_lr, target_names=target_names))

    #hyperparameter search
    print("=========Hyperparameter search=========")
    
    model_RR=RandomForestClassifier()
    tuned_parameters = {'min_samples_leaf': range(10,100,10), 'n_estimators' : range(10,100,10),'max_features':['auto','sqrt','log2']}
    RR_model= RandomizedSearchCV(model_RR, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1)
    RR_model.fit(X_train, y_train)
    print(f'Mejor estimador: ', RR_model.best_params_)

    y_prob = RR_model.predict_proba(X_test)[:,1] 
    y_pred = np.where(y_prob > 0.5, 1, 0) 
    print(f'score: ', RR_model.score(X_test, y_pred))

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



if __name__ == '__main__':
    main()