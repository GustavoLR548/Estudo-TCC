import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def create_neural_networks():
    classificador = Sequential()

    camada_escondida = Dense(units = 4,
                            activation = 'relu',
                            input_dim = 4)

    camada_escondida2 = Dense(units = 4,
                            activation = 'relu')

    """
    Diferente da classificação binária, neste caso temos 
    três neurônios na camada de saída, pois temos três possíveis
    resultados. Além disso, a nossa função de ativação será a softmax, pois
    queremos transformar um vetor de dados (neste caso as informações processadas
    das petalas), em um valor número probabilístico.
    """
    camada_de_saida = Dense(units = 3, 
                            activation = 'softmax')

    classificador.add(camada_escondida)
    classificador.add(camada_escondida2)
    classificador.add(camada_de_saida)

    classificador.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["categorical_accuracy"])

    return classificador


base       = pd.read_csv("iris.csv")
previsores = base.iloc[:,0:4].values
classe     = base.iloc[:,4].values

# Transformar os dados em String para tipo numérico
label_enconder = LabelEncoder()
classe = label_enconder.fit_transform(classe)

# Transformar o array de uma dimensão em três dimensões, para se ajustar 
# com o modelo de saida que tem três possibilidades
classe_dummy = np_utils.to_categorical(classe)

classificador = KerasClassifier(build_fn=create_neural_networks,
                                epochs=1000,
                                batch_size=10)

resultados = cross_val_score(estimator=classificador,
                            X=previsores,
                            y=classe,
                            cv=10,
                            scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()

print(media)
print(desvio)