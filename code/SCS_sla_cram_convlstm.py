from keras import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D
from CRAM_ConvLSTM import CRAM_ConvLSTM

channels   = 3
WIDTH      = 80
HEIGHT     = 80
ker        = 3
batch_size = 32
epochs     = 2000
ReduceLR_patience=30

# Model
input_data = Input(shape=(None,HEIGHT, WIDTH, channels))
x = input_data
output_data = CRAM_ConvLSTM(input_data,HEIGHT,WIDTH,channels) 
# -------------
model = Model(input_data, output_data)
model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-8), loss='logcosh', metrics=['mae'])
model.summary()

