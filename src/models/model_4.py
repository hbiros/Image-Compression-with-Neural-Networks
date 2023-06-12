from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Model
import keras

# from pupieapp_metric.pu_pieapp_metric import PUPieAppMetric


pool_factor = (2,2) 
img_dim = 128

input_layer = Input(shape=(64, 64, 3), name="input_layer")


conv_1 = Conv2D(16, (3,3), activation='relu', padding='same', name='conv_1')(input_layer)
pool_1 = MaxPooling2D(pool_factor, name='pool_1')(conv_1) # 32x32x16
conv_2 = Conv2D(8, (3,3), activation='relu', padding='same', name='conv_2')(pool_1) # 32 x 32 x 8
pool_2 = MaxPooling2D(pool_factor, name='pool_2')(conv_2) # 16x16x8
conv_2 = Conv2D(8, (3,3), activation='relu', padding='same', name='conv_3')(pool_2) # 32 x 32 x 8
pool_2 = MaxPooling2D(pool_factor, name='pool_3')(conv_2) # 16x16x8
flatten_layer = Flatten()(pool_2)

dense_1 = Dense(256, activation='relu')(flatten_layer)


code_layer = Dense(img_dim, activation='relu', name='code_layer')(dense_1)

dense_5 = Dense(256, activation='relu')(code_layer)

reshape_layer = Reshape((8, 8, 4))(dense_5)
unconv_layer_01 = Conv2DTranspose(8, (3,3), activation='relu', padding='same', name='unconv_layer_01')(reshape_layer)
upsamp_layer_02 = UpSampling2D(pool_factor, name='upsamp_layer_02')(unconv_layer_01)
unconv_layer_1 = Conv2DTranspose(8, (3,3), activation='relu', padding='same', name='unconv_layer_1')(upsamp_layer_02)
upsamp_layer_2 = UpSampling2D(pool_factor, name='upsamp_layer_2')(unconv_layer_1)
unconv_layer_2 = Conv2DTranspose(16, (3,3), activation='relu', padding='same', name='unconv_layer_2')(upsamp_layer_2)
upsamp_layer_3 = UpSampling2D(pool_factor, name='upsamp_layer_3')(unconv_layer_2)


from keras import backend as K

def custom_activation(x):
    return K.sigmoid(x/10)

output_layer = Conv2D(3, (3,3), padding='same', activation=custom_activation, name='output_layer')(upsamp_layer_3)


model = Model(input_layer, output_layer)

opt = keras.optimizers.Adam(learning_rate=0.001)

# model.compile(optimizer=opt, loss='mse', run_eagerly=True, metrics=[PUPieAppMetric()])
model.compile(optimizer=opt, loss='mse', run_eagerly=True)

encoder = Model(inputs=model.input, outputs=model.get_layer('code_layer').output)

model.summary()