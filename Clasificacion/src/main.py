import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets, ensemble
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score, f1_score, accuracy_score,roc_curve,classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score,LeaveOneOut
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
import copy

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

def split_data(X, Y, train_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    X_train: pd.DataFrame
    Dataset de entrenamiento
    X_test: pd.DataFrame
    Dataset de prueba
    """
    return train_test_split(X, Y, train_size=train_size)

def logistic_regression(X, y) -> LogisticRegression:
    """
    Esta función se encarga de entrenar el modelo logistico
    
    Parámetros
    -----------
    X: np.ndarray
    Matriz de datos de entrada

    Y: np.ndarray
    Matriz de datos de salida

    Returns
    -----------
    logireg: LogisticRegression
    Regresor logistico
    """
    #Creacion del regresor logistico 
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)

    # Entrenamiento
    logireg.fit(X, y)

    return logireg

def svm_regression(X, y) -> SVC:
    """
    Esta función se encarga de entrenar el modelo SVM
    
    Parámetros
    -----------
    X: np.ndarray
    Matriz de datos de entrada

    y: np.ndarray
    Matriz de datos de salida

    Returns
    -----------
    svc: SVC
    Regresor SVM
    """
    #Creacion del regresor SVM
    svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)

    # Entrenamiento 
    svc.fit(X, y)

    return svc

def compare_models(X, y):
    """
    Esta función se encarga de comparar los modelos entrenados
    """
    particions = [0.5, 0.7, 0.8]

    for part in particions:
        #Separamos los datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = split_data(X, y, part)

        #Entrenamos el modelo logistico
        logireg = logistic_regression(X_train, y_train)
        print ("Correct classification Logistic: ", part, "% of the data: ", logireg.score(X_test, y_test))

        #Entrenamos el modelo SVM
        svc = svm_regression(X_train, y_train)
        probs = svc.predict_proba(X_test)
        print ("Correct classification SVM: ", part, "% of the data: ", svc.score(X_test, y_test))

    return probs, y_test

def generate_precision_recall(X, y, n_classes: int, probs: np.array, y_v):

    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])

        plt.plot(recall[i], precision[i],
        label='Precision-recall curve of class {0} (area = {1:0.2f})'
                            ''.format(i, average_precision[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")
        plt.title('Precision-Recall curve')
    plt.show()

def generate_roc_curve(n_classes: int, probs: np.array, y_v):
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.legend()
    plt.show()

def make_meshgrid(x, y, h=.02):
    """
    Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """
    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def show_C_effect(C=1.0, gamma=0.7, degree=3):
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = data[:, :2]
    y = dataset['output']

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    #C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1= X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Frecuencia cardíaca máxima alcanzada')
        ax.set_ylabel('Tipo de dolor en el pecho')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

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


def dataset_atipic_values(dataset):
    fig, ax = plt.subplots(ncols = 7, nrows = 2, figsize = (20, 10))
    index = 0
    ax = ax.flatten()

    for col, value in dataset.items():
        sns.boxplot(y=col, data=dataset, ax=ax[index])
        index += 1
    plt.tight_layout(pad = 0.5, w_pad=0.7, h_pad=5.0)
    plt.title("Distribución de los valores atípicos")
    plt.show()

def normalization(dataset):
    dataset.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def add_category_values(dataset):
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

def one_hot_encoding(dataset):
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

def train_model(model, X_train, X_test, y_train, y_test, model_name):
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

def k_split(X, y):
    #Usemos cross_val_score para evaluar una puntuación mediante validación cruzada.
    kf =KFold(n_splits=5, shuffle=True, random_state=42)
    cnt = 1
    # split()  method generate indices to split data into training and test set.
    for train_index, test_index in kf.split(X, y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        cnt += 1

    return kf

def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')

def model_estimator(X, y, kf):
    #A partir de 250 es un buen estimador !
    estimators = [50, 100, 150, 200, 250]

    for count in estimators:
        score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= count, random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")
        print(f'For estimators: {count}')
        rmse(score.mean())

def model_fit(X, y, kf):
    n_estimators = [50, 100, 150, 200, 250, 300]

    for val in n_estimators:
        score = cross_val_score(ensemble.RandomForestClassifier(n_estimators= val, random_state= 42), X, y, cv= kf, scoring="accuracy")
        print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

def leave_one_out(model_rf, X, y, method_score, splits):
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    crossvalidation = KFold(n_splits=splits, random_state=None, shuffle=False)
    scores = cross_val_score(model_rf, X, y, scoring=method_score, cv=crossvalidation, n_jobs=1)
    print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))))

def evaluate_preds(y_true, y_preds):
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    print(f"Acc : {round(accuracy, 2) * 100:.2f}%")
    print(f"Precision : {round(precision, 2):.2f}")
    print(f"recall : {round(recall, 2):.2f}")
    print(f"F1 score {round(f1, 2):.2f}")

def simple_roc_curve(y_test, y_preds, title):
    fpr, tpr, threshold = roc_curve(y_test, y_preds)
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
    #comparamos modelos
    X =  dataset[['thalachh','cp']].values
    y = dataset['output']

    prob, y_v = compare_models(X, y)
    generate_precision_recall(X,y, 2, prob, y_v)
    generate_roc_curve(2, prob, y_v)
    #mIRAR fUNCIONES QUE LUIS NO ESTA USANDO, ESTA EL CODIGO AQUI

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

    print("=========Regression SVC(rbf)=========")
    model_svc = SVC(kernel='rbf', random_state = 42)
    model_svc, valor, y_pred_svc = train_model(model_svc, X_train, X_test, y_train, y_test, "SVC(rbf)")
    modelos = modelos.append(valor, ignore_index=True)

    print("=========Regression SVC(linear)=========")
    model_svc2 = SVC(kernel='linear', random_state = 42)
    model_svc2, valor, y_pred_svc2 = train_model(model_svc2, X_train, X_test, y_train, y_test, "SVC(linear)")
    modelos = modelos.append(valor, ignore_index=True)

    print("=========Regression SVC(poly)=========")
    model_svc3 = SVC(kernel='poly', random_state = 42)
    model_svc3, valor, y_pred_svc3 = train_model(model_svc3, X_train, X_test, y_train, y_test, "SVC(poly)")
    modelos = modelos.append(valor, ignore_index=True)

    print("=========Random forest=========")
    model_rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=0.16, random_state=42)
    model_rf, valor, y_pred_rf = train_model(model_rf, X_train, X_test, y_train, y_test, "Random forest")
    modelos = modelos.append(valor, ignore_index=True)

    print("=========KNN=========")
    model_knn = KNeighborsClassifier(n_neighbors=17)
    model_knn, valor, y_pred_knn = train_model(model_knn, X_train, X_test, y_train, y_test, "KNN")
    modelos = modelos.append(valor, ignore_index=True)

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
    model_estimator(X, y, kf)

    print("=========Evaluamos de la regresion logistica=========")
    score = cross_val_score(LogisticRegression(), X, y, cv= kf, scoring="neg_mean_squared_error")
    print(f'Scores para cada fold son: {score}')
    rmse(score.mean())

    print("=========Crossvalidaion en base al acuracy=========")
    print("=========Evaluamos del Random forest=========")
    score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores para cada fold son: {score}')
    print(f'Average score: {"{:.2f}".format(score.mean())}')
    model_fit(X, y, kf)

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

    print("=========Classificacion report random forest=========")
    target_names = ['Padece de enfermedad', 'No padece de enfermedad']
    print(classification_report(y_test, y_pred_rf, target_names=target_names))

    print("=========Classificacion report regression logistica=========")
    print(classification_report(y_test, y_pred_lr, target_names=target_names))

if __name__ == '__main__':
    main()