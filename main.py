import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head())
    print(dt_heart.shape)
    
    # Eliminamos el target, para obtener solo las caracteristicas
    dt_features = dt_heart.drop(['target'], axis=1)
    
    # tomamos la columna target
    dt_target = dt_heart['target']
    
    
    print(dt_features.shape)
    print(dt_target.shape)
    
    # Para PCA necesitamos normalizar los datos con standarScaler lo podemows hacer 
    dt_features = StandardScaler().fit_transform(dt_features)
    
    # partimos el conjunto de entranamiento
    # la partición se hace de manera aleatoria,
    #   si quiero que no sea asi, debo usar el parametro random_state=numeroCualquiera,
    #   mientras sea el mismo numero la particion será igual
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30)
    
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)
    
    #default n_components = min(n_muestras, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)
    
    # manda los datos de a poco, ideal para maquinas con pocos recursos
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()
    
    # Regresión logistica
    
    # solver='lbgfgs' => evita errores y advertencia
    logistic = LogisticRegression(solver='lbfgs')
    
    # aplicamos el algoritmo
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA:", logistic.score(dt_test, y_test))
    
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)     
    print("SCORE IPCA:", logistic.score(dt_test, y_test))
    
                          
    
    
    