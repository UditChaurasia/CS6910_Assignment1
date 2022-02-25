import math
import numpy as np
from matplotlib import pyplot
from keras.datasets import fashion_mnist
# loading the data
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
trainX = trainX/255
testX = testX/255

# Same as that for Question 2 and 3 but here a minor change is we added a confusion matrix and updated it accordingly


def layer_init(number_of_layers, layer_input):
  for layer in range (0, number_of_layers):
    if (wt_initialisation == 'random'):
      if (layer == 0):
        weight_matrix_of_layers.append(np.random.normal(0,0.5,(layer_input[layer],784)))
        bias_matrix_of_layers.append(np.random.normal(0,0.5,(layer_input[layer],1)))
      elif (layer == (number_of_layers - 1)):
        weight_matrix_of_layers.append(np.random.normal(0,0.5,(10,layer_input[layer - 1])))
        bias_matrix_of_layers.append(np.random.normal(0,0.5,(10,1)))
      else:
        weight_matrix_of_layers.append(np.random.normal(0,0.5,(layer_input[layer],layer_input[layer - 1])))
        bias_matrix_of_layers.append(np.random.normal(0,0.5,(layer_input[layer],1)))



def logistic(val):
  temp = []
  if (activation_function == 'tanh'):
    for i in range(0,len(val)):
      if (val[i] > 0):
        try:
          x = math.exp(2*val[i])
        except OverflowError:
          x = 1
      elif (val[i] <= 0):
        try:
          x = math.exp(-2*val[i])
        except OverflowError:
          x = -1

      if (x == 1 or x == -1):
        temp.append(x)
      else:
        temp.append((math.exp(2*val[i])-1)/(math.exp(2*val[i])+1))

        

  elif (activation_function == 'relu'):
    for i in range(0,len(val)):

      

      if (val[i] <= 0):
        temp.append(0.0)
      else:
        temp.append(val[i])

  elif (activation_function == 'sigmoid'):
    for i in range(0,len(val)):

      


      if (val[i] > 35):
        temp.append(1.0)
      elif (val[i] < -35):
        temp.append(0.0)
      else:
        x = (1/(1 + math.exp(-val[i])))
        temp.append(x)

  temp = np.array(temp)
  return temp




# This function implements the final output function, which is softmax in our case
def soft_max(temp_w):
  temp = []
  denominator = 0
  for i in range (0, len(temp_w)):
    denominator += math.exp(temp_w[i])
  for i in range (0, len(temp_w)):
    temp.append(math.exp(temp_w[i])/denominator)
  return temp

# This function implements the loss function that has to be minimized
def loss_function (var):
  return -math.log(var)


def forward_propagation(input_vector):
  activation_output[0] = input_vector
  for i in range(0,number_of_layers - 1):
    activation_input[i] = np.dot(weight_matrix_of_layers[i],activation_output[i]) + bias_matrix_of_layers[i]

    activation_output[i+1] = logistic(activation_input[i])
    activation_output[i+1] = activation_output[i+1].reshape(layer_input[i],1)
  activation_input[number_of_layers - 1] = np.matmul(weight_matrix_of_layers[number_of_layers - 1],activation_output[number_of_layers - 1]) + bias_matrix_of_layers[number_of_layers - 1]
  activation_output[number_of_layers] = soft_max(activation_input[number_of_layers - 1])
  activation_output[number_of_layers] = np.array(activation_output[number_of_layers])
  activation_output[number_of_layers] = activation_output[number_of_layers].reshape(10,1)
  activation_output[number_of_layers] = np.array(activation_output[number_of_layers])
  return activation_output[number_of_layers]


def backward_propagation(y):
  grad_a = []
  grad_h = []
  grad_wt = []
  grad_bias = []
  one_hot_vector = np.zeros(10)
  one_hot_vector = one_hot_vector.reshape(10,1)
  one_hot_vector[y] = 1
  grad_a.append(-(one_hot_vector - activation_output[number_of_layers]))
  ind = 0
  for i in range (number_of_layers - 1, -1, -1):
    grad_wt.append(np.dot(grad_a[ind],np.transpose(activation_output[i])))
    grad_bias.append(grad_a[ind])
    grad_h.append(np.dot(np.transpose(grad_wt[ind]),grad_a[ind]))
    if (i != 0):
      temp = []
      for j in range (0, layer_input[i-1]):
        var = activation_output[i][j]
        temp.append((grad_h[ind][j])*var*(1 - var))
      temp = np.array(temp)
      temp = temp.reshape(layer_input[i-1],1)
      grad_a.append(temp)
    ind += 1
  return (grad_wt, grad_bias)




def fwd_pro(input_vector):
  test_activation_output[0] = input_vector
  for i in range(0,number_of_layers - 1):
    test_activation_input[i] = np.dot(weight_matrix_of_layers[i],test_activation_output[i]) + bias_matrix_of_layers[i]
    test_activation_output[i+1] = logistic(test_activation_input[i])
    test_activation_output[i+1] = test_activation_output[i+1].reshape(layer_input[i],1)
  test_activation_input[number_of_layers - 1] = np.matmul(weight_matrix_of_layers[number_of_layers - 1],test_activation_output[number_of_layers - 1]) + bias_matrix_of_layers[number_of_layers - 1]
  test_activation_output[number_of_layers] = soft_max(test_activation_input[number_of_layers - 1])
  test_activation_output[number_of_layers] = np.array(test_activation_output[number_of_layers])
  test_activation_output[number_of_layers] = test_activation_output[number_of_layers].reshape(10,1)
  return test_activation_output[number_of_layers]


def trainD(layer_inputX, wt_initialisationX, activation_functionX, epochsX, etaX, optimizerX, betaX, epsilonX, beta1X, beta2X, batch_sizeX):
  global activation_function
  global epochs
  global eta
  global optimizer
  global beta
  global epsilon
  global beta1
  global number_of_layers
  global layer_input
  global test_activation_input
  global test_activation_output
  global weight_matrix_of_layers
  global bias_matrix_of_layers
  global activation_input
  global activation_output
  global wt_initialisation
  global beta2
  global batch_size
  global confusion_matrix

  confusion_matrix = []
  for i in range(0,10):
    temp = [0 for j in range(10)]
    confusion_matrix.append(temp) 

  true_label = ['l1','l2','l3','l4','l5','l6','l7','l8','l9','l10']
  predicted_label = ['pl1','pl2','pl3','pl4','pl5','pl6','pl7','pl8','pl9','pl10']
  print("done")


  batch_size = batch_sizeX
  beta2 = beta2X
  gwt_initialisation = 'random'
  test_activation_input = []
  test_activation_output = []
  weight_matrix_of_layers = []
  bias_matrix_of_layers = []
  activation_input = []
  activation_output = []

  
  layer_input = layer_inputX
  wt_initialisation = wt_initialisationX
  activation_function = activation_functionX
  epochs = epochsX
  eta = etaX
  optimizer = optimizerX
  beta = betaX
  epsilon = epsilonX
  beta1 = beta1X
  number_of_layers = len(layer_input) + 1
  for i in range(0, number_of_layers):
    activation_input.append(None)
    test_activation_input.append(None)
  
  for i in range(0, number_of_layers + 1):
    activation_output.append(None)
    test_activation_output.append(None)

  layer_init(number_of_layers, layer_input)

  loss = 0
  for ep in range (0, epochs):
    tempW = []
    tempB = []
    tempW1 = []
    tempB1 = []
    print("Epoch Number:", ep + 1)
    for data in range (0,54000,batch_size):
      if ((data + 1) % 5000 == 0):
        print(data + 1)
      x = trainX[data]
      x = x.ravel()
      x = x.reshape(784,1)
      y = trainY[data]

      temp_loss_vector = forward_propagation(x)
      loss += -np.log(temp_loss_vector[y])


      
      (gradW,gradB) = backward_propagation(y)
      if (optimizer == 'momentum'):
        gradW = np.array(gradW)
        gradB = np.array(gradB)
        for i in range (0, number_of_layers):
          if (data == 0):
            weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - eta*gradW[number_of_layers - 1 -i]
            bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - eta*gradB[number_of_layers - 1 -i]
            tempW.append(gradW[number_of_layers - 1 -i])
            tempB.append(gradB[number_of_layers - 1 -i])
          else:
            tempW = np.array(tempW)
            tempB = np.array(tempB)
            weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - eta*gradW[number_of_layers - 1 -i] - beta*tempW[i]
            bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - eta*gradB[number_of_layers - 1 -i] - beta*tempB[i]
            tempW[i] = eta*gradW[number_of_layers - 1 -i] + beta*tempW[i]
            tempB[i] = eta*gradB[number_of_layers - 1 -i] + beta*tempB[i]

      elif(optimizer == 'sgd'):
        for i in range (0, number_of_layers):
          weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - eta*gradW[number_of_layers - 1 -i]
          bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - eta*gradB[number_of_layers - 1 -i]


      elif(optimizer == 'nestrov'):
        for i in range (0, number_of_layers):
          weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - eta*gradW[number_of_layers - 1 -i]
          bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - eta*gradB[number_of_layers - 1 -i]
        
        if(data+1 < 53999):
          (gW,gB) = backward_propagation(trainY[data+1])

        for i in range (0, number_of_layers):
          weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - eta*gW[number_of_layers - 1 -i] - beta*gradW[number_of_layers - 1 -i]
          bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - eta*gB[number_of_layers - 1 -i] - beta*gradB[number_of_layers - 1 -i]

      elif(optimizer == 'rmsprop'):
        gradW = np.array(gradW)
        gradB = np.array(gradB)
        tt = 0
        for i in range (0, number_of_layers):
          if (data == 0):
            weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - (eta/np.sqrt(epsilon))*gradW[number_of_layers - 1 -i]
            bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - (eta/np.sqrt(epsilon))*gradB[number_of_layers - 1 -i]
            tempW.append((1 - beta1)*np.square(gradW[number_of_layers - 1 -i]))
            tempB.append((1 - beta1)*np.square(gradB[number_of_layers - 1 -i]))
          
            
          else:
            tempW = np.array(tempW)
            tempB = np.array(tempB)
            weight_matrix_of_layers[i] -= (eta/np.sqrt(epsilon + tempW[tt]))*gradW[number_of_layers - 1 -i]
            bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - (eta/np.sqrt(epsilon + tempB[tt]))*gradB[number_of_layers - 1 -i]
            
            tempW[tt] = beta1*tempW[tt] + (1 - beta1)*np.square(gradW[number_of_layers - 1 -i])
            tempB[tt] = beta1*tempB[tt] + (1 - beta1)*np.square(gradB[number_of_layers - 1 -i])
            tt += 1

      elif(optimizer == 'adam'):
        for i in range (0, number_of_layers):
          if (data == 0):
            weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - (eta/np.sqrt(epsilon))*gradW[number_of_layers - 1 -i]
            bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - (eta/np.sqrt(epsilon))*gradB[number_of_layers - 1 -i]
            tempW.append((1 - beta1)*gradW[number_of_layers - 1 -i])
            tempB.append((1 - beta1)*gradB[number_of_layers - 1 -i])
            tempW1.append((1 - beta2)*np.square(gradW[number_of_layers - 1 -i]))
            tempB1.append((1 - beta2)*np.square(gradB[number_of_layers - 1 -i]))
          else:
            tempW = np.array(tempW)
            tempB = np.array(tempB)            
            tempW1 = np.array(tempW1)
            tempB1 = np.array(tempB1)
            tempW[i] = tempW[i]/(1 - (beta1)**data)
            tempB[i] = tempB[i]/(1 - (beta1)**data)
            tempW1[i] = tempW1[i]/(1 - (beta2)**data)
            tempB1[i] = tempB1[i]/(1 - (beta2)**data)          
            weight_matrix_of_layers[i] = weight_matrix_of_layers[i] - (eta/np.sqrt(epsilon + tempW1[i]))*tempW[i]
            bias_matrix_of_layers[i] = bias_matrix_of_layers[i] - (eta/np.sqrt(epsilon + tempB1[i]))*tempB[i]
            tempW[i] = beta1*tempW[i] + (1 - beta1)*gradW[number_of_layers - 1 -i]
            tempB[i] = beta1*tempB[i] + (1 - beta1)*gradB[number_of_layers - 1 -i]
            tempW1[i] = beta2*tempW1[i] + (1 - beta2)*np.square(gradW[number_of_layers - 1 -i])
            tempB1[i] = beta2*tempB1[i] + (1 - beta2)*np.square(gradB[number_of_layers - 1 -i])
  accuracy = 0
  for test in range (0,10000):
      x = testX[test]
      x = x.ravel()
      x = x.reshape(784,1)
      res = fwd_pro(x)
      maxV = 0
      maxI = 0
      for i in range(0,10):
        if (maxV <= res[i]):
          maxV = res[i]
          maxI = i
      if (maxI == testY[test]):
        accuracy += 1
        print(accuracy)
        confusion_matrix[maxI][testY[test]] += 1

      wandb.log({'Confusion Matrix': wandb.plots.HeatMap(true_label,predicted_label,confusion_matrix,show_text = True)})

     
    
  


!pip install wandb 
import wandb
wandb.login
wandb.init(project = 'Assignment1', entity = 'uditchaurasia', reinit = True)
trainD([128,96,128],'random','sigmoid',4,0.001,'nestrov',0.05,0.01,0.05,0.05,64)
