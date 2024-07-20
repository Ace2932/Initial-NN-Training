
import tensorflow as tf
import pandas as pd
import numpy as np


df = pd.read_csv("auto-mpg.csv")
df = df[["mpg", "horsepower", "model year"]]

df['horsepower'].replace('?', np.nan, inplace=True)
df['horsepower'] = df['horsepower'].astype(float)

# Remove rows with missing values
df.dropna(subset=['horsepower'], inplace=True)

horsepowerList = df["horsepower"].to_list()

print(horsepowerList)

horsepowerNP = np.array(horsepowerList, dtype = float)


mpgString = df["mpg"].to_list()
mpgNP = np.array(mpgString, dtype = float)

modelYearString = df["model year"]
modelYearNP = np.array(modelYearString, dtype = float)

for i,c in enumerate(horsepowerNP):
  print("{} horsepower = {} miles per gallon".format(c, mpgNP[i]))

# concatanates the two 
inputsNP = np.column_stack((horsepowerNP, modelYearNP))

# Define the model using the functional API
inputs = tf.keras.Input(shape=(2,))
outputs = tf.keras.layers.Dense(units=1)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# l0 = tf.keras.layers.Dense(units=1, input_shape=[1])


model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.009))

history = model.fit(inputsNP, mpgNP, epochs=500, verbose=False)
print("Model Finished Training....")

horsepower = 97.0
model_year = 70
test = np.array([[horsepower, model_year]], dtype=float)

print(model.predict(test))
