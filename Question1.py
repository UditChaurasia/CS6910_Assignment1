from matplotlib import pyplot
from keras.datasets import fashion_mnist
# load dataset
(trainX, trainY), (testX, testy) = fashion_mnist.load_data()
# summarize loaded dataset
count = 0
temp = [11]
for i in range(0,60000):
  if (count == 10):
    break
  else:
    if (trainY[i] not in temp):
      temp.append(trainY[i])
      print("Image corresponding to the label : ", trainY[i])
      pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
      pyplot.show()
