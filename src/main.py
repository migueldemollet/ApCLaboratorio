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


def main():
    # Apartado C
    dataset, dataset_test = load_db('db\\Data_train_reduced.csv', 'db\\Data_train_reduced.csv')
    prepare_data(dataset)
    prepare_data(dataset_test)
    dataset_statistics(dataset)
    dataset_graphics(dataset)

if __name__ == '__main__':
    main()