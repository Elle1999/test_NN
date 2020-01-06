import numpy as np

class Perceptron():
  #creates the 
  
  def __init__(self, input_shape):
    self.weights = np.random.randint() * np.random.random((input_shape , 1))

  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

  def forward(self, x):
    weighted_sum = np.dot(x, self.weights)
    activated_output = sigmoid(weighted_sum)
    return activated_output
  
  def backward(self, x, y):
    activated_output = self.forward(x)
    error = y - activated_output
    updates = error * sigmoid_derivative(activated_output)
    self.weights += np.dot(x.T, updates)
  
  def train(self, x, y, epochs=10000):
    for i in range(epochs):
      for input in x:
        predicted = self.forward(x)
        actual = y
        
        if i % 1000 == 0:
          print("Predicted: ", predicted)
          print("Actual: ", actual)
        
        self.backward(x, y)
