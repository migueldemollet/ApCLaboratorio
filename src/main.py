from sklearn import linear_model
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import scipy.stats
import seaborn as sns
import copy

def load_db(dir_db: str, dir_test_db: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Obligamos que meutre todas las columnas
    pd.set_option('display.max_columns', None)

    # Visualitzarem només 3 decimals per mostra
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Funcio per a llegir dades en format csv
    def load_dataset(path):
        dataset = pd.read_csv(path, header=0, delimiter=',')
        return dataset

    # Carreguem dataset d'exemple
    dataset = load_dataset(dir_db)
    dataset_test = load_dataset(dir_test_db)
    
    return dataset, dataset_test

def columns_with_na(dframe: pd.DataFrame) -> int:
    temp = dframe.isna().sum()
    temp = temp[temp >0]
    print(f"Columns containing nan values:{temp.index}")
    return temp.index

def prepare_data(dataset: pd.DataFrame) -> None:
    if (dataset.isnull().sum().any() > 0):
        columns_to_drop = list(columns_with_na(dataset))
        dataset.drop(columns_to_drop, axis=1, inplace = True)
        sns.heatmap(dataset.isnull(), cbar=False)

def dataset_statistics(dataset: pd.DataFrame) -> None:
    print("=========Descripcion=========")
    print(dataset.describe(), end="\n")

    print("=========Atributos=========")
    print(dataset.keys(), end="\n")

    print("=========Los 5 primeros elementos=========")
    print(dataset.head(), end="\n")

    print("=========Total de Productos=========")
    print(dataset['Product'].value_counts())
    print("Total = ", len(dataset))

def dataset_graphics(dataset: pd.DataFrame) -> None:
    #Media
    dataset['mean']=(dataset.mean(axis=1)/(dataset.shape[1]-3))
    plt.figure()
    plt.title("Valoracion media por producto")
    plt.bar(dataset['Product'],dataset['mean'])
    plt.show()
    # Correlacion
    correlacio = dataset.iloc[:,3:10].corr()
    fig , ax = plt.subplots()
    sns.heatmap(correlacio,annot=True)
    plt.show()
    #dataset histogramas
    dataset.hist(figsize=(50,50))
    plt.show()

def split_data(dataset: pd.DataFrame, dataset_test: pd.DataFrame, objective: str):

    X_train=dataset.values[:,np.newaxis,3]

    #Defino los datos correspondientes a las opinion de desodorante
    y_train=dataset[objective]
    #Separamos los datos de "train" en entrenamiento y prueba
    return train_test_split(X_train,y_train,test_size=0.2,random_state=15)

def regresion_lineal(X_train: list , X_test: list , y_train: list ,y_test:list, x_label: str):
    #definimos el algoritmo a utilizar
    lr = linear_model.LinearRegression()

    #Entreno de modelo
    lr.fit(X_train, y_train)

    #Realizar una prediccion
    Y_pred = lr.predict(X_test)

    #Graficamos los datos junto con el modelo
    plt.scatter(X_test,y_test)
    plt.plot(X_test,Y_pred,color='red',linewidth=3)
    plt.title('Regresion Lineal Simple')
    plt.xlabel(x_label)
    plt.ylabel('Opinion personal de desodorante')
    plt.show()

    print("Mean squeared error: ", mean_squared_error(X_train, y_train))
    print("R2 Score: ", lr.score(X_train, y_train))

def standardization(dataset: pd.DataFrame):
    to_drop=['Respondent.ID','Product.ID','Product']
    dataset.drop(to_drop, axis=1, inplace = True)
    dataset=dataset.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    dataset=(dataset - dataset.min()) / ( dataset.max() - dataset.min())

def cleanData (dataDrop):
    to_drop=['q1_1.personal.opinion.of.this.Deodorant','Respondent.ID','Product.ID','Product']
    dataDrop.drop(to_drop, inplace=True, axis=1)
    return dataDrop.values

def regresion_multiple(X_multiple,y_multiple):
    #Separamos los datos de "train" en entrenamiento y prueba
    X_train,X_test,y_train,y_test = train_test_split(X_multiple,y_multiple, test_size=0.2)

    #definimos el algoritmo a utilizar
    lr_multiple = linear_model.LinearRegression()

    #Entreno de modelo
    lr_multiple.fit(X_train,y_train)

    #Realizar una prediccion
    Y_pred_multiple = lr_multiple.predict(X_test)

    print('Datos del modelo de regresion multiple')
    print('--------------------------------------')
    print('Valor de las pendientes o coeficiente "a":')
    print(lr_multiple.coef_)
    print()
    print('Valor de las interseccion de  "b":')
    print(lr_multiple.intercept_)
    print()
    print('Precision del algoritmo:')
    print(lr_multiple.score(X_train,y_train))


def main():
    # Apartado C
    dataset, dataset_test = load_db('db\\Data_train_reduced.csv', 'db\\Data_train_reduced.csv')
    prepare_data(dataset)
    prepare_data(dataset_test)
    dataset_statistics(dataset)
    dataset_graphics(dataset)

    # Apartado B
    print("==========Primera Regresion Lineal=========")
    X_train,X_test,y_train,y_test = split_data(dataset, dataset_test, 'q1_1.personal.opinion.of.this.Deodorant')
    regresion_lineal(X_train, X_test, y_train, y_test, "valoracion de atractivo")
    
    print("=========Segunda Regresion Lineal=========")
    #La diferencia que hay respeto la anterior es la columna en la que deseamos predecir
    X_train,X_test,y_train,y_test = split_data(dataset, dataset_test, 'q9.how.likely.would.you.be.to.purchase.this.Deodorant')
    regresion_lineal(X_train, X_test, y_train, y_test, "Gusto Instantaneo de desodorante")

    print("=========Regresion Lineal con la Normalizacion=========")
    dataset_norm = copy.deepcopy(dataset)
    standardization(dataset_norm)
    X_train,X_test,y_train,y_test = split_data(dataset_norm, dataset_test, 'q1_1.personal.opinion.of.this.Deodorant')
    regresion_lineal(X_train, X_test, y_train, y_test, "valoracion de atractivo")

    print("=========Primera Regresion Lineal Multiple=========")
    y_multiple=dataset['q1_1.personal.opinion.of.this.Deodorant'].values
    data = cleanData(dataset)
    regresion_multiple(data[:,0:], y_multiple)

    print("=========Segunda Regresion Lineal Multiple=========")
    #La diferencia que hay respeto la anterior es que no estamos usando el atributo Instant.Liking
    regresion_multiple(data[:,1:], y_multiple)

if __name__ == '__main__':
    main()