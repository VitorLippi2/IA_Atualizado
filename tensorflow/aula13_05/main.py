import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = "tensorflow/auto-mpg/auto-mpg.data"

columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(
    url, 
    names=columns,
    na_values='?', 
    comment='\t',
    sep=r'\s+', 
    skipinitialspace=True
)

print(raw_dataset.info())

dataset = raw_dataset.copy()
dataset = dataset.dropna()

origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

print(dataset)