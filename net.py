from numpy import array, dot, exp, random
import mnist

# 58 is my favorite number
random.seed(58)

def sigmoid(x):
  return 1 / (1 + exp(-x))

def arc_sigmoid(x):
  return x * (1 - x)

class Layer():
  def __init__(self, num_nodes, num_inputs):
    self.weights = 2 * random.random((num_inputs, num_nodes)) - 1

  def print_weights(self):
    print(self.weights)

class NNet():
  def __init__(self, layer1, layer2, layer3):
    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3 

  def train(self, input_data, labels, num_epochs):
    for _ in range(num_epochs):
      layer1_output, layer2_output, layer3_output = self.generate_layer_output(input_data)
 
      # back propogation steps 
      layer3_error = labels - layer3_output
      layer3_delta = layer3_error * arc_sigmoid(layer3_output)

      layer2_error = dot(layer3_delta, self.layer3.weights.T)
      layer2_delta = layer2_error * arc_sigmoid(layer2_output)

      layer1_error = dot(layer2_delta, self.layer2.weights.T)
      layer1_delta = layer1_error * arc_sigmoid(layer1_output)

      # update weights after calculating the deltas
      self.layer1.weights += dot(input_data.T, layer1_delta)
      self.layer2.weights += dot(layer1_output.T, layer2_delta)
      self.layer3.weights += dot(layer2_output.T, layer3_delta)

  def generate_layer_output(self, input_data):
    layer1_output = sigmoid(dot(input_data, self.layer1.weights))
    layer2_output = sigmoid(dot(layer1_output, self.layer2.weights))
    layer3_output = sigmoid(dot(layer2_output, self.layer3.weights))
    return layer1_output, layer2_output, layer3_output
  
  def predict(self, input_data):
    # generate the layer outputs, only keep last output
    return self.generate_layer_output(input_data)[2]

  def print_net(self):
    print("Layer 1 weights:")
    self.layer1.print_weights()
    print("Layer 2 weights:")
    self.layer2.print_weights()
    print("Layer 3 weights:")
    self.layer3.print_weights()

# 16 nodes with 784 inputs
layer1 = Layer(16, 784)
# 16 nodes with 16 inputs
layer2 = Layer(16, 16)
# 10 nodes with 16 inputs
layer3 = Layer(10, 16)

perceptron = NNet(layer1, layer2, layer3)

x_train, t_train, x_test, t_test = mnist.load()

# train for 5k epochs
#perceptron.train(train_input, train_labels, 5000)

print(perceptron.predict(x_train[0]))
