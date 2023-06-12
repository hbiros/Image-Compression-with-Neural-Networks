from image_processing.prepare_data import load_data
from image_processing.visualisation import show_data
import click
import numpy as np 
import pickle
from datetime import datetime
from keras.callbacks import CSVLogger


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
@click.option(
              '-s',
              '--save',
              prompt='Save model', 
              is_flag=True
              )
def train(model, epochs, batch_size, save):

  if(model == 1):
    from models.model_1 import model, encoder 
  elif(model == 2):
    from models.model_2 import model, encoder
  elif(model == 3):
    from models.model_3 import model, encoder
  else:
    raise ValueError('Model {} does not exits'.format(model))  
  
  x_test, x_train = load_data(train_data="data/cat_faces_train", test_data="data/cat_faces_test")
  
  time = datetime.now().strftime("%H_%M_%S")
  log_name = 'log_' + time + '.csv'
  csv_logger = CSVLogger(log_name, append=True, separator=',')
  
  history = model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test), callbacks=[csv_logger])
  
  encoded_cats = encoder.predict(x_test)
  encoded_cats = encoded_cats.reshape(len(encoded_cats),-1)
  print(encoded_cats.shape)
  reconstructed_cats = model.predict(x_test)
  np.clip(reconstructed_cats, 0, 1)
  print(reconstructed_cats.shape)
  show_data(x_test, reconstructed_cats)

  if save:
    model_name="model_"+time
    model_history="history_"+time
    model.save(model_name)
    with open(model_history, 'wb') as file_pi:
      pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    train()