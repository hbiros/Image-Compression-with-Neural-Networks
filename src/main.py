import sys
import numpy as np
from PIL import Image
from image_processing.prepare_data import printProgressBar
import keras
from pupieapp_metric.pu_pieapp_metric import PUPieAppMetric

def crop_image(image, chunk_size):
    width, height = image.size
    crop_width = (width // chunk_size) * chunk_size
    crop_height = (height // chunk_size) * chunk_size
    return image.crop((0, 0, crop_width, crop_height))

def split_and_merge_image(image_name):
    with Image.open(image_name) as image:
        chunk_size = 64

        cropped_image = crop_image(image, chunk_size)

        fragments = []
        for j in range(0, cropped_image.height, chunk_size):
            for i in range(0, cropped_image.width, chunk_size):
                fragment = cropped_image.crop((i, j, i + chunk_size, j + chunk_size))
                fragments.append(np.array(fragment))

        fragments_array = np.array(fragments)
        length = fragments_array.shape[0]
        reconstructed = []
        model = keras.models.load_model("./saved_model", custom_objects={"PUPieAppMetric": PUPieAppMetric()})
        
        for i, fragment in enumerate(fragments_array):
            printProgressBar(i+1, length, prefix="Progress", suffix="Complete", length=100)
            reconstructed.append(model.predict(fragment))

        output_width = cropped_image.width
        output_height = cropped_image.height
        merged_image = Image.new('RGB', (output_width, output_height))

        for i, fragment in enumerate(reconstructed):
            col = i % (output_width // chunk_size)
            row = i // (output_width // chunk_size)
            fragment_image = Image.fromarray(fragment)
            merged_image.paste(fragment_image, (col * chunk_size, row * chunk_size))

        output_name = image_name.replace('.jpg', '_reconstructed.jpg')
        merged_image.save(output_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the image name as an argument.")
    else:
        image_name = sys.argv[1]
        split_and_merge_image(image_name)
