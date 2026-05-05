import random

class Perceptron:
    def __init__(self, num_weights):
        self.weights = [random.uniform(-1, 1) for _ in range(num_weights)]
        # taxa de aprendizado - influencia no quanto varia a linha de ajusta (botão do rádio)
        self.lr = 0.005 # 1 fica mto alto, ele muda muito a linha

    def guess(self, inputs): # inputs = pontos de entrada. No nosso caso, 100 pts.
        total = sum(inputs[i] * self.weights[i] for i in range(len(self.weights)))
        return self._sign(total)

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr

    def guess_y(self,x):
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]

        return -(w2/w1) - (w0/w1) * x

    def _sign(self, num): # função de ativação
        return 1 if num >= 0 else -1