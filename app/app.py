import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
# Cargar el conjunto de datos en un DataFrame de Pandas
df = pd.read_csv('./imdb_movies.csv')

# presupuesto
X = df[['budget_x']]

X = (X - X.mean()) / X.std()

# ganancia
y = df['revenue']

# Normalizar los datos
y = (y - y.mean()) / y.std()

# datos de entrenamiento.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# creacion de la red neuronal.
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(1))


# compilar la red neunoral.
model.compile(loss='mse', optimizer='adam')

# entrenarla
model.fit(X, y, epochs=100, batch_size=32)

# datos de prueba
test = model.evaluate(X_test, y_test, batch_size=32)
print("conjunto de prueba: ", test)

# ganancia para una película con presupuesto de $100 millones
prediction = model.predict(pd.DataFrame({'Budget': [100]}))
print("Ganancia predicha para un presupuesto de $100 millones: ", prediction[0][0] * y.std() + y.mean())
