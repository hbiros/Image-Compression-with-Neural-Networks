from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
import keras

from pupieapp_metric.pu_pieapp_metric import PUPieAppMetric


pool_factor = (2,2)

input_layer = Input(shape=(64, 64, 3), name="input_layer")
conv_1 = Conv2D(16, (3,3), activation='relu', padding='same', name='conv_1')(input_layer)
pool_1 = MaxPooling2D(pool_factor, name='pool_1')(conv_1)
conv_2 = Conv2D(8, (3,3), activation='relu', padding='same', name='conv_2')(pool_1)
pool_2 = MaxPooling2D(pool_factor, name='pool_2')(conv_2)
conv_3 = Conv2D(8, (3,3), activation='relu', padding='same', name='conv_3')(pool_2)

code_layer = MaxPooling2D(pool_factor, name='code_layer')(conv_3)

unconv_layer_0 = Conv2DTranspose(8, (3,3), padding='same', activation='relu', name='unconv_layer_0')(code_layer)
upsamp_layer_1 = UpSampling2D(pool_factor, name='upsamp_layer_1')(unconv_layer_0)
unconv_layer_1 = Conv2DTranspose(8, (3,3), activation='relu', padding='same', name='unconv_layer_1')(upsamp_layer_1)
upsamp_layer_2 = UpSampling2D(pool_factor, name='upsamp_layer_2')(unconv_layer_1)
unconv_layer_2 = Conv2DTranspose(16, (3,3), activation='relu', padding='same', name='unconv_layer_2')(upsamp_layer_2)
upsamp_layer_3 = UpSampling2D(pool_factor, name='upsamp_layer_3')(unconv_layer_2)

from keras import backend as K

def custom_activation(x):
    return K.sigmoid(x/10)

output_layer = Conv2D(3, (3,3), padding='same', activation=custom_activation, name='output_layer')(upsamp_layer_3)

model = Model(input_layer, output_layer)

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

encoder = Model(inputs=model.input, outputs=model.get_layer('code_layer').output)