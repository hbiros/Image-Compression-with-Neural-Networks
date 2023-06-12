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
@click.option(
              '-t',
              '--train_data',
              prompt='Train dataset', 
              default="data/train_fragments",
              type=str
              )
@click.option(
              '-r',
              '--ratio',
              prompt='Split ratio', 
              default=0.1, 
              type=float
              )
def train(model, epochs, batch_size, save, train_data, ratio):

  if(model == 1):
    from models.model_1 import model, encoder 
  elif(model == 2):
    from models.model_2 import model, encoder
  elif(model == 3):
    from models.model_3 import model, encoder
  elif(model == 4):
    from models.model_4 import model, encoder
  else:
    raise ValueError('Model {} does not exits'.format(model))  
  
  x_train = load_data(train_data=train_data)
  
  time = datetime.now().strftime("%H_%M_%S")
  log_name = 'log_' + time + '.csv'
  csv_logger = CSVLogger(log_name, append=True, separator=',')
  
  history = model.fit(x_train, x_train, epochs=epochs, shuffle=True, batch_size=batch_size, validation_split=ratio, callbacks=[csv_logger])
  
  encoded_fragments = encoder.predict(x_train)
  encoded_fragments = encoded_fragments.reshape(len(encoded_fragments),-1)
  
  reconstructed_fragments = model.predict(x_train)
  np.clip(reconstructed_fragments, 0, 1)

  show_data(x_train, reconstructed_fragments)

  if save:
    model_name="model_"+time
    model_history="history_"+time
    model.save(model_name)
    with open(model_history, 'wb') as file_pi:
      pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    train()