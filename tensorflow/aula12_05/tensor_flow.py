import tensorflow as tf

# conjunto de treinamento
xs = tf.constant([0,1,2,3,4], dtype=tf.float32)
ys = xs * 1.2 + 5 # função definida/ escolhida. Pega os valores do tensor e soma

# print(xs)
# print(ys) - só pro commit

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

print("Treinando")
model.compile(loss="mean_squared_error", optimizer='sgd')
model.fit(xs, ys, epochs=500)

def linear_reg():
    x_max = 20
    x_arr = []
    y_arr = []
    correct_arr = []

    for x in range(10, x_max + 1):
        x_tensor = tf.constant([float(x)], dtype=tf.float32) # transforma o x passado em um tensor
        y_pred = model.predict(x_tensor, verbose=0) # previsão só aceita tensor, por isso a linha acima
        x_arr.append(x) # entrada (x)
        y_arr.append(float(y_pred[0][0])) # saída prevista pelo tensor
        correct_arr.append(float(x * 1.2 + 5)) # saída correta

    display(x_arr, y_arr, correct_arr)

def display(x_arr, y_arr, correct_arr):
    text = "Correct                Predicted \n"
    
    for i in range(len(x_arr)):
        correct = correct_arr[i]
        predicted = y_arr[i]
        text += f"{correct:.4f}                {predicted:.4f}\n"
    print(text)

linear_reg()