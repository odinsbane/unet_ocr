from tensorflow import keras
import sys

class ModelBuilder:
    """
        This builds the detection model. Its a pretty standard unet
    """
    def build( self ):
        inp_l = keras.layers.Input((240, 320, 3))
        l_i = inp_l
        filters = 16
        
        across = []
        depth = 4
        for d in range(depth):
            l_i = keras.layers.Conv2D(filters, (3, 3), padding="same")(l_i)
            filters *= 2
            l_i = keras.layers.Conv2D( filters, (3, 3), padding="same")(l_i)
            across.append(l_i)
            l_i = keras.layers.MaxPool2D((2,2))(l_i)
        
        l_i = keras.layers.Conv2D(filters, (3, 3), padding="same")(l_i)
        filters *= 2
        l_i = keras.layers.Conv2D( filters, (3, 3), padding="same")(l_i)
        
        for d in range(depth):
            l_i = keras.layers.Conv2DTranspose(filters, (3, 3), (2, 2), padding="same")(l_i)
            l_i = keras.layers.Concatenate()( [ l_i, across[-(d+1)] ])
            l_i = keras.layers.Conv2D(filters, (3, 3), padding="same")(l_i)
            filters /= 2
            l_i = keras.layers.Conv2D( filters, (3, 3), padding="same")(l_i)
        op1 = keras.layers.Conv2D( 4, (1, 1), padding="same")(l_i)
        op2 = keras.layers.Conv2D( 1, (1, 1), padding="same")(l_i)
        return keras.models.Model(inputs=[inp_l], outputs = {"boxes":op1, "score":op2})
        
if __name__=="__main__":
    if len(sys.argv) < 2:
        print("usage: mode.py model_name")
        sys.exit(0)
        
    builder = ModelBuilder()
    model = builder.build()
    model.summary()
    model.save(sys.argv[1])
