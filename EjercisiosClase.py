import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import openpyxl

# pip install openpyxl pandas scikit-learn numpy

df = pd.read_excel('C:/Users/s4ds/Documents/DolarData.xlsx')

X = df[['Día']]
y = df['ValorDólar']

model = LinearRegression()
model.fit(X, y)
prediccion = model.predict(pd.DataFrame({'Día': [8]}))

print(f"Predicción del valor del dólar para mañana (X=8): {prediccion[0]}")