import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = "../auto-mpg/auto-mpg.data"

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

# dummies
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

#remove 20% do dataset (amostra aleatória) para treinar
train_dataset = dataset.sample(frac = 0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe().transpose()[["mean", "std"]] #std=desvop padrão
print(train_stats)

# normalização
def norm(x):
    return ( x - train_stats["mean"]) / train_stats["std"]

train_dataset = norm(train_dataset)
test_dataset = norm(test_dataset)

print(dataset)