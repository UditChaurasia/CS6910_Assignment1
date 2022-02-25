from matplotlib import pyplot
from keras.datasets import fashion_mnist
def img_plot():
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
        image = wandb.Image(trainX[i], caption="Label Image")
        wandb.log({"Label": image})
      

      
!pip install wandb
import wandb
wandb.login

wandb.init(project = 'Assignment1', entity = 'uditchaurasia', reinit = True)
img_plot()
