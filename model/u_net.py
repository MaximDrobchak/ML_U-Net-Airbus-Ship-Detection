from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D,  BatchNormalization, Conv2DTranspose

def get_unet_model(input_shape=(256, 256, 3), num_classes=1):

    def fire(x, filters, kernel_size):
        y1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
        y2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(y1)
        y3 = BatchNormalization(momentum=0.9)(y2)     
        return y3

    def fire_module(filters, kernel_size):
        return lambda x: fire(x, filters, kernel_size)

    def fire_up(x, filters, kernel_size, concat_layer):
        y1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        y2 = concatenate([y1, concat_layer])
        y3 = fire_module(filters, kernel_size)(y2)
        return y3

    def up_fire_module(filters, kernel_size, concat_layer):
        return lambda x: fire_up(x, filters, kernel_size, concat_layer)

    input_img = Input(shape=input_shape) #256

    down1 = fire_module(8, (3, 3))(input_img)
    pool1 = MaxPooling2D((2, 2))(down1)  #128

    down2 = fire_module(16, (3, 3))(pool1)
    pool2 = MaxPooling2D((2, 2))(down2) #64
    
    down3 = fire_module(32, (3, 3))(pool2)
    pool3 = MaxPooling2D((2, 2))(down3) #32
    
    down4 = fire_module(64, (3, 3))(pool3)
    pool4 = MaxPooling2D((2, 2))(down4) #16
    
    down5 = fire_module(128, (3, 3))(pool4) #center
    
    up6 = up_fire_module(64, (3, 3), down4)(down5) #32
    up7 = up_fire_module(32, (3, 3), down3)(up6) #64
    up8 = up_fire_module(16, (3, 3), down2)(up7) #128
    up9 = up_fire_module(8, (3, 3), down1)(up8) #256
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(up9)

    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  

    return model