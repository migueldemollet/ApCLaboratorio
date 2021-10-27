from sklearn import linear_model
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
from numpy import mean
from numpy import std

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

def load_db(dir_db: str, dir_test_db: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esta función se encarga de leer los ficheros csv y transformarlo en variables de tipo pd.DataFrame gracias a la ayuda 
    de la librería de pandas
    
    Parámetros
    -----------
    Dir_db: str
    String que indica el directorio de la base de datos para el entreno del modelo
    
    Dir_test_db: str
    String que indica el directorio de la base de datos para el test del modelo
    
    Returns
    -----------
    pd.DataFrame: dataframe del directorio de entreno
    pd.DataFrame: dataframe del directorio de test
    """
    # Visualitzarem només 3 decimals per mostra
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    dataset = pd.read_csv(dir_db, header=0, delimiter=',')
    dataset_test = pd.read_csv(dir_test_db, header=0, delimiter=',')
    
    return dataset, dataset_test

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
    print("=========Descripcion=========")
    print(dataset.describe(), end="\n")

    print("=========Atributos=========")
    print(dataset.keys(), end="\n")

    print("=========Los 5 primeros elementos=========")
    print(dataset.head(), end="\n")

    print("=========Total de Productos=========")
    print(dataset['Product'].value_counts())
    print("Total = ", len(dataset), end="\n")

def dataset_graphics(dataset: pd.DataFrame) -> None:
    """
    Funcion similar a la anteriormente nombrada, pero en este caso realiza gráficos sobre la media, 
    correlación entre las variables e histogramas de los atributos
    
    Paraemtros
    -----------
    dataset: pd.DataFrame
    dataset que queremos explorer
    
    Returns
    -----------
    Ninguno
    """
    #Media
    dataset['mean']=(dataset.mean(axis=1)/(dataset.shape[1]-3))
    plt.figure()
    plt.title("Valoracion media por producto")
    plt.bar(dataset['Product'],dataset['mean'])
    plt.show()

    # Correlacion
    correlacio = dataset.iloc[:,3:50].corr()
    fig , ax = plt.subplots()
    fig.set_figwidth(40)
    fig.set_figheight(40)
    sns.heatmap(correlacio,annot=True,cmap="YlGnBu")

    # Dataset histogramas
    dataset.hist(figsize=(50,50))
    plt.show()

def split_data(dataset: pd.DataFrame, dataset_test: pd.DataFrame, objective: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Esta función la usamos para separar los datos de entreno con los datos de test en una proporción de 80% entreno y 20% de test
    
    Parámetros
    -----------
    dataset: pd.DataFrame
    dataset del cual se establrecera una X y una Y de entreno
    
    dataset: pd.DataFrame
    dataset del cual se establecerá una X y una Y de test
    
    objective: str
    string el cual elige el atributo objetivo a predecir
    
    Returns
    -----------
    np.ndarray: numpy array en que su contenido se encuentra la X de entreno
    np.ndarray: numpy array en que su contenido se encuentra la X de test
    np.ndarray: numpy array en que su contenido se encuentra la Y de entreno
    np.ndarray: numpy array en que su contenido se encuentra la Y de test
    """
    X_train=dataset.values[:,np.newaxis,3]
    y_train=dataset[objective]

    return train_test_split(X_train,y_train,test_size=0.2,random_state=15)

def regresion_lineal(X_train: list , X_test: list , y_train: list ,y_test:list, x_label: str) -> None:
    """
    Función encargada de crear y entrenar un modelo con una regresión lineal, y su graficacion es decir graficar 
    los datos junto al modelo 
    
    Parámetros
    -----------
    X_train: list
    Listas de los datos el cual el modelo usara para entrenar
    
    X_test: list
    Listas de los datos el cual el modelo usara para testear
    
    y_train: list
    Listas de los datos el cual el modelo debe de predecir según los datos de entrada (X_train)
    
    y_test:list
    Listas de los datos el cual el modelo debe de predecir según los datos de entrada (X_test)
    
    x_label: str
    String que se usa como titulo del eje x de la grafica
    
    Returns
    -----------
    Ninguno
    """
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
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

def standardization(dataset: pd.DataFrame) -> None:
    """
    Función que normaliza con el método de la medida un dataframe y elimina los atributos 
    necesarios para la normalizacion
    
    Parámetros
    -----------
    dataset: pd.dataframe
    dataframe a normalizar
    
    Returns
    -----------
    Ninguno
    """
    to_drop=['Respondent.ID','Product.ID','Product']
    dataset.drop(to_drop, axis=1, inplace = True)

    dataset=dataset.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    dataset=(dataset - dataset.min()) / ( dataset.max() - dataset.min())

def cleanData (dataDrop: pd.DataFrame) -> np.ndarray:
    """
    Función que elimina los atributos innecesarios por ejemplo Product.ID, Product, etc. para poder realizar la 
    regresión lineal múltiple.
    
    Parámetros
    -----------
    dataDrop: pd.DataFrame
    dataframe que queremos limpiar
    
    Returns
    -----------
    np.ndarray: Dataframe limpio
    """
    to_drop=['q1_1.personal.opinion.of.this.Deodorant','Respondent.ID','Product.ID','Product']
    dataDrop.drop(to_drop, inplace=True, axis=1)

    return dataDrop.values

def regresion_multiple(X_multiple: np.ndarray ,y_multiple: np.ndarray) -> None:
    """
    Función encargada de separar los datos de entreno y los de test, crear y entrenar un modelo con una regresión 
    lineal multiple, y mostrar por terminal su precisión de predicción
    
    Parámetros
    -----------
    X_multiple: np.ndarray
    Atributos que se usaran para predecir
    
    y_multiple: np.ndarray
    Atributos a predecir
    
    Returns
    -----------
    Ninguno
    """
    X_train, X_test, y_train, y_test = train_test_split(X_multiple,y_multiple, test_size=0.2)
    
    lr_multiple = linear_model.LinearRegression()
    lr_multiple.fit(X_train,y_train)
    lr_multiple.predict(X_test)

    print('Datos del modelo de regresion multiple')
    print('--------------------------------------')
    print('Valor de las pendientes o coeficiente "a":')
    print(lr_multiple.coef_, end="\n")

    print('Valor de las interseccion de  "b":')
    print(lr_multiple.intercept_, end="\n")

    print('Precision del algoritmo:')
    print(lr_multiple.score(X_train,y_train), end="\n")

def apply_pca(dataset: pd.DataFrame, X_train: np.ndarray, y_train: np.ndarray, components: int) -> tuple[PCA, np.ndarray]:
    """
    Funcion que se encarga de aplicar el PCA (análisis de componentes principales) y printar por terminal métricas de rendimiento y 
    resultado del nuevo dataset como por ejemplo el MSE, cross validation, etc.
    
    Parámetros
    -----------
    dataset: pd.DataFrame
    dataset el cual le aplicaremos el PCA
    
    X_train: np.ndarray
    Atributos de entrada que usaremos para la el cross validation y MSE
    
    y_train: np.ndarray
    Atributos de salida que usaremos para la el cross validation y MSE
    
    components: int
    dimensión a la que queremos reducir
    
    Returns
    -----------
    PCA: Matriz que contiene los datos del dataset reducido a la dimensión indicada
    np.ndarray: Matriz del PCA transpuesta
    """
    X = dataset.values.T
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')

    pca = PCA(n_components=components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Defenimos el pipeline
    steps = [('pca', PCA(n_components=1)), ('m', LogisticRegression())]
    model = Pipeline(steps=steps)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))    
    print("Mean squeared error: ", mean_squared_error(X_train, y_train))
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    return pca, X_pca

def show_pca(pca: PCA, X_pca, dataset: pd.DataFrame) -> None:
    """
    Esta funcion muestra de forma grafica lo que el PCA hace a un dataset
    
    Parámetros
    -----------
    pca: PCA
    Matriz resultante de la funcion apply_pca
    
    X_pca
    Matriz PCA transpuesta
    
    dataset: pd.DataFrame
    dataset original sin reducer
    
    Returns
    -----------
    Ninguno
    """
    X_new = pca.inverse_transform(X_pca)
    plt.scatter(dataset.values[:, 0], dataset.values[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()

def variance(dataset: pd.DataFrame, y_multiple: np.ndarray) -> None:
    """
    Función que grafica la varianza individual y acumulada de las variables y a mas vuelve a setear los valores 
    de personal opinión que fueron anteriormente eliminados
    
    Parámetros
    -----------
    dataset: pd.DataFrame
    dataframe el cual usaremos sus atributos para calcular la varianza
    
    y_multiple: np.ndarray
    valores del atributo personal opinión
    
    Returns
    -----------
    Ninguno
    """
    # Vuelvo añadir la columna 'q1_1.personal.opinion.of.this.Deodorant'
    dataset['q1_1.personal.opinion.of.this.Deodorant'] = y_multiple
    X_std = dataset.values[:,0:53]
    
    # Calculamos los autovalores y autovectores de la matriz y los mostramos
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Hacemos una lista de parejas (autovector, autovalor) 
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Ordenamos estas parejas den orden descendiente con la función sort
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # A partir de los autovalores, calculamos la varianza explicada
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    with plt.style.context('seaborn-pastel'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(1,53), var_exp[:52], alpha=0.5, align='center',label='Varianza individual explicada', color='g')
        plt.step(range(53), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
        plt.ylabel('Ratio de Varianza Explicada')
        plt.xlabel('Componentes Principales')
        plt.legend(loc='best')
        plt.tight_layout()

    plt.show()


def gradient_descent(X: np.ndarray ,y: np.ndarray ,learning_rate: float, iterations: int) -> tuple[float, float, float]:
    """
    Función que aplica nuestra versión del descenso del gradiante
    
    Parámetros
    ----------
    X: np.ndarray
    Parametros de entrada
    
    y: np.ndarray
    parametros de salida
    
    learning_rate: float
    learning rate
    
    iterations: int
    variable que indica las iteraciones del descenso de gradiante
    
    Returns
    -----------
    float: resultado de la formula de la recta m*x + b
    float: parámetro b de la fórmula de la recta
    float: parámetro m de la fórmula de la recta
    """
    b = 0
    m = 5
    n = X.shape[0]
    for _ in range(iterations):
        # MSE
        b_gradient = -2 * np.sum(y - (m*X + b))/n
        m_gradient = -2 * np.sum(X*(y - (m*X + b)))/n
        
        # Actualizando ultimos valores
        b = b - (learning_rate * b_gradient)
        m = m - (learning_rate * m_gradient)

        # Actualizando valores previos
        b = b - (learning_rate * b_gradient)
        m = m - (learning_rate * m_gradient)
    return m*X + b,b,m

def show_gradient_descent(X: np.ndarray, Y: np.ndarray, Xgraf: float) -> None:
    """
    funcion que grafica que el modelo resultante al aplicar el descenso de gradiante
    
    Parámetros
    -----------
    X: np.ndarray
    Datos para la gráfica en eje x
    
    Y: np.ndarray
    Datos para la gráfica en eje y
    
    Xgraf: float
    Formula de la recta resultante de aplicar el descenso de gradiante
    
    Returns
    -----------
    Ninguno
    """
    plt.style.use('fivethirtyeight')
    plt.scatter(X, Y, color='black')
    plt.plot(X,Xgraf)
    plt.gca().set_title("Gradient Descent Linear Regressor")
    plt.show()

def test_gradient_descent(X: np.ndarray, Y: np.ndarray) -> None:
    """
    función que realiza varias veces el descenso de gradiante para poder probar como se comporta delante varios aprendizajes
    
    Parámetros
    -----------
    X: np.ndarray
    Parametros de entrada
    
    Y: np.ndarray
    Parametros de salida
    
    Returns
    -----------
    Ninguno
    """
    learnig=[0.1,0.2,0.001]
    iterations=[1000,2000]
    for i in learnig:
        for j in iterations:
            Xgraf,c,m=gradient_descent(X,Y,i,j)

            plt.style.use('fivethirtyeight')
            plt.scatter(X, Y, color='black')
            plt.plot(X,Xgraf)
            plt.gca().set_title("Gradient Descent Linear Regressor with multiple values")
        
    plt.show()

def polinomial_regressor(data: pd.DataFrame, grade: int) -> None:
    """
    Funcion encargada de separar los datos de un dataset (recibido por parametro), en entreno y test en una proporcion de
    80% entreno y 20% test, el cual se usara para realizar la regresion polinomial

    Parametros
    -----------
    data: pd.DataFrame
    dataset el cual se usara para realizar la regresion polinomial
    
    grade: int
    grado del polinomio

    Returns
    -----------
    Ninguno
    """
    X_p=data[:,np.newaxis,3]
    y_p=data[:,4]
    X_train_p,X_test_p,y_train_p,y_test_p=train_test_split(X_p,y_p, test_size=0.2)

    poli_reg = PolynomialFeatures(degree=grade)

    #Calculando los valores de X para cada una de las potencias 
    X_train_poli=poli_reg.fit_transform(X_train_p)
    X_test_poli=poli_reg.fit_transform(X_test_p)

    #Definir el algoritmo
    pr=linear_model.LinearRegression()

    pr.fit(X_train_poli,y_train_p)
    
    #realizo
    Y_pred_pr=pr.predict(X_test_poli)

    plt.scatter(X_test_p,y_test_p)
    plt.plot(X_test_p,Y_pred_pr,color='black',linewidth=3)
    plt.title('Regresion Polinomial')
    plt.xlabel('Gusto Instantaneo de desodorante')
    plt.ylabel('Opinion personal de desodorante')
    plt.show()

    print('Precision del modelo:')
    print(pr.score(X_train_poli,y_train_p))

def main():
    # Apartado C
    dataset, dataset_test = load_db('db\\Data_train_reduced.csv', 'db\\test_data.csv')
    dataset_norm = copy.deepcopy(dataset)
    dataset_descent = copy.deepcopy(dataset)

    prepare_data(dataset)
    prepare_data(dataset_test)
    dataset_statistics(dataset)
    dataset_graphics(dataset)

    # Apartado B
    print("==========Primera Regresion Lineal=========")
    X_train, X_test, y_train, y_test = split_data(dataset, dataset_test, 'q1_1.personal.opinion.of.this.Deodorant')
    regresion_lineal(X_train, X_test, y_train, y_test, "valoracion de atractivo")
    
    print("=========Segunda Regresion Lineal=========")
    # La diferencia que hay respeto la anterior es la columna en la que deseamos predecir
    X_train,X_test,y_train,y_test = split_data(dataset, dataset_test, 'q9.how.likely.would.you.be.to.purchase.this.Deodorant')
    regresion_lineal(X_train, X_test, y_train, y_test, "Gusto Instantaneo de desodorante")

    print("=========Regresion Lineal con la Normalizacion=========")
    standardization(dataset_norm)
    X_train,X_test,y_train,y_test = split_data(dataset_norm, dataset_test, 'q1_1.personal.opinion.of.this.Deodorant')
    regresion_lineal(X_train, X_test, y_train, y_test, "valoracion de atractivo")

    print("=========Primera Regresion Lineal Multiple=========")
    y_multiple=dataset['q1_1.personal.opinion.of.this.Deodorant'].values
    data = cleanData(dataset)
    regresion_multiple(data[:,0:], y_multiple)

    print("=========Segunda Regresion Lineal Multiple=========")
    # La diferencia que hay respeto la anterior es que no estamos usando el atributo Instant.Liking
    regresion_multiple(data[:,1:], y_multiple)

    print("=========PCA=========")
    pca, X_pca = apply_pca(dataset, X_train, y_train, 10) 
    show_pca(pca, X_pca, dataset)
    variance(dataset, y_multiple)

    # Apartado A
    print("=========Descenso de Gradiante=========")
    plt.rcParams['figure.figsize'] = (10, 7.0)
    data=dataset_descent.values
    X = data[:,3]
    Y = data[:,4]
    Xgraf,c,m=gradient_descent(X, Y, 0.1, 1000)

    show_gradient_descent(X, Y, Xgraf)

    test_gradient_descent(X, Y)

    print("=========Regresion Polinomial=========")
    polinomial_regressor(data, 2)

if __name__ == '__main__':
    main()