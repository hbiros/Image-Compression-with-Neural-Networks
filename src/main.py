import numpy as np
from PIL import Image
import click
import keras
# from pupieapp_metric.pu_pieapp_metric import PUPieAppMetric

@click.command()
@click.option(
              '-m',
              '--model',
              prompt='Model to load', 
              required=True, 
              type=str
              )
@click.option(
              '-i',
              '--img_name',
              prompt='Image to reconstruct', 
              required=True, 
              type=str
              )
def split_and_merge_image(model, img_name):

    with Image.open(img_name) as image:   
        
        chunk_size = 64
        width, height = image.size
        
        chunks_per_width = (width // chunk_size)
        crop_width = chunks_per_width * chunk_size
        
        chunks_per_height = (height // chunk_size)
        crop_height = chunks_per_height * chunk_size
        
        cropped_image = image.crop((0, 0, crop_width, crop_height))
        fragments = []

        for i in range(0, cropped_image.width, chunk_size):
            for j in range(0, cropped_image.height, chunk_size):
                fragment = cropped_image.crop((i, j, i + chunk_size, j + chunk_size))
                fragments.append(np.array(fragment))


        fragments = np.array(fragments)
        fragments = fragments / 255.0

        model = keras.models.load_model(model)
        
        reconstructed = model.predict(fragments)

        reconstructed = np.uint8(reconstructed * 255)

        output_width = crop_width
        output_height = crop_height
        merged_image = Image.new('RGB', (output_width, output_height))

        index = 0
        for i in range(0, cropped_image.width, chunk_size):
            for j in range(0, cropped_image.height, chunk_size):
                fragment_image = Image.fromarray(np.uint8(reconstructed[index]*255))
                merged_image.paste(fragment_image, (i, j))
                index += 1

        output_name = img_name.replace('.jpg', '_reconstructed.jpg')
        merged_image.save(output_name)

if __name__ == "__main__":
    split_and_merge_image()