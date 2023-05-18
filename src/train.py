from models.model_1 import model, encoder
from image_processing.prepare_data import load_data
from image_processing.visualisation import show_data

if __name__ == "__main__":
  x_test, x_train = load_data(train_data="data/cat_faces_train", test_data="data/cat_faces_test")
  model.fit(x_train, x_train, epochs=10, batch_size=32, validation_data=(x_test, x_test))
  encoded_cats = encoder.predict(x_test)
  encoded_cats = encoded_cats.reshape(len(encoded_cats),-1)
  print(encoded_cats.shape)
  reconstructed_cats = model.predict(x_test)
  print(reconstructed_cats.shape)
  show_data(x_test, reconstructed_cats)
  ans = input("Save model? (y/n): ").upper()
  if ans == 'Y':
    model.save('.')