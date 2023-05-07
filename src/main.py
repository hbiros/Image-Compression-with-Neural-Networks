from models.model_1 import model, encoder
from image_processing.prepare_data import load_data
from image_processing.visualisation import show_data

if __name__ == "__main__":
  x_test, x_train = load_data(train_data="data/cat_faces_train", test_data="data/cat_faces_test")
  model.fit(x_train, x_train, epochs=20, batch_size=32, validation_data=(x_test, x_test))
