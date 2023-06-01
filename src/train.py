from image_processing.prepare_data import load_data
from image_processing.visualisation import show_data
import click
import numpy as np 

@click.command()
@click.option(
              '-m',
              '--model',
              prompt='Model to train', 
              default=1, 
              type=int
              )
@click.option(
              '-e',
              '--epochs',
              prompt='Number of epochs', 
              default=10, 
              type=int
              )
@click.option(
              '-bs',
              '--batch_size',
              prompt='Batch size', 
              default=32, 
              type=int
              )
def train(model, epochs, batch_size):

  if(model == 1):
    from models.model_1 import model, encoder 
  elif(model == 2):
    from models.model_2 import model, encoder
  else:
      raise ValueError('Model {} does not exits'.format(model))  
  
  model.summary()

  x_test, x_train = load_data(train_data="data/cat_faces_train", test_data="data/cat_faces_test")
  model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))
  encoded_cats = encoder.predict(x_test)
  encoded_cats = encoded_cats.reshape(len(encoded_cats),-1)
  print(encoded_cats.shape)
  reconstructed_cats = model.predict(x_test)
  np.clip(reconstructed_cats, 0, 1)
  print(reconstructed_cats.shape)
  show_data(x_test, reconstructed_cats)
  ans = input("Save model? (y/n): ").upper()
  if ans == 'Y':
    model.save('.')


if __name__ == "__main__":
    train()