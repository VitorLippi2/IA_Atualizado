# printar quantos acertos/ taxa de acertos tivemos
# printar pesos testados

# exercícios:
# altere a taxa de aprendizado
# 

"""
P: altere a quantidade de pontos desenhados de 100 para 500 - ele aprende melhor ou fica mais lento para processar?
R: Não. Pontos que estão longe da reta não influenciam, apenas os que estão próximos à reta. O que influencia é a qualidade dos pontos inseridos (mais próximos à reta), não a quantidade.

P: Modifique a função (fx) no arquivo point.py para:
y = -2x + 0.5
y = 0.8x - 0.3

Verifique se o Perceptron consegue se adaptar a qualquer inclinação e deslocamento (bias)

(coloque a função pra passar no ponto *0,0* para analisar o comportamento do bias)


Suponha que x seja o peso de uma laranja e y seja a acidez. Laranjas do tipo A têm y > 0.5x + 0.1 e tipo B o restante.
Modifique o arquivo point.py para refletir essa regra.
Coloque os valores de uma nova laranja e imprima o tipo dela.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from perceptron import Perceptron
from point import Point, f

def setup():
    global perceptron,points,training_index, treshold
    perceptron = Perceptron(3)
    points = [Point(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(100)]
    training_index = [0]
    treshold = 0.99

fig, ax = plt.subplots(figsize=(6, 6))

def draw_line():
    p1x = -1
    p1y = f(-1)

    p2x = 1
    p2y = f(1)
    ax.plot([p1x, p2x], [p1y, p2y], color="purple", linewidth=1)

    gp1x = -1
    gp1y = perceptron.guess_y(-1)
    gp2x = 1
    gp2y = perceptron.guess_y(1)
    ax.plot([gp1x, gp2x], [gp1y, gp2y], color="blue", linewidth=1)


def train_single_point():
    pt = points[training_index[0]]
    inputs = [pt.x, pt.y, pt.bias]
    perceptron.train(inputs, pt.label)
    training_index[0] = (training_index[0] + 1) % len(points)

def draw(frame):
    ax.clear()
    # Define o intervalo fixo de -1 a 1
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    # Adiciona linhas de grade e eixos centrais para clareza
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    acertos = 0

    # Desenhar pontos
    for pt in points:
        face_color = 'black' if pt.label == 1 else 'white'
        ax.plot(pt.x, pt.y, 'o', color='black', markerfacecolor=face_color,
                markersize=10, markeredgewidth=1, zorder=2)

        guess = perceptron.guess([pt.x, pt.y, pt.bias])
        color = 'green' if guess == pt.label else 'red'
        if (guess == pt.label): acertos += 1
        ax.plot(pt.x, pt.y, 'o', color=color, markersize=5, zorder=3)

    
    draw_line()
    if ((acertos/len(points)) < treshold): 
        train_single_point()
    else:
        print(f'Parou o treinamento. \n Pesos w0: {perceptron.weights[0]} \n Pesos w1: {perceptron.weights[1]} \n Pesos w2: {perceptron.weights[2]} ')
    ax.set_title("Perceptron: Classificação no intervalo [-1, 1]")

setup()

ani = animation.FuncAnimation(fig, draw, interval=10, cache_frame_data=False)
plt.show()

