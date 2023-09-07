"""
 Given a bounding box and an image, how can we say what the text it.
"""
from tensorflow import keras
import numpy

CIPHER=b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.\" "
NULL=len(CIPHER) - 1
import data
import sys
import math

class Caligrapher:
    def __init__(self):
        self.input_size = (16, 96)
        self.max_characters = 12
        self.n = len(CIPHER) 
    def buildModel(self):
        inp = keras.layers.Input( (self.input_size[0], self.input_size[1], 3) )
        x = inp
        x = keras.layers.Conv2D(8, (3, 3) )(x)
        x = keras.layers.Conv2D(8, (3, 3) )(x)
        x = keras.layers.Conv2D(16, (3, 3))(x)
        x = keras.layers.Conv2D(16, (3, 3) )(x)
        x = keras.layers.Conv2D(32, (3, 3) )(x)
        x = keras.layers.Conv2D(32, (3, 3))(x)
        total = x.shape.dims[1].value * x.shape.dims[2].value * x.shape.dims[3].value
        x = keras.layers.Reshape((total,))(x)
        x = keras.layers.Dense(100)(x);
        x = keras.layers.Dense(100)(x);
        x = keras.layers.Dense((self.max_characters*self.n))(x)
        x = keras.layers.Reshape( (self.max_characters, self.n))(x)
        x = keras.activations.softmax(x)
        print("total:",x.shape)
        return keras.models.Model(inputs=[inp], outputs=[x])
        
    def encodeString(self,  truth ):
        """
            This is a string of n characters long. 
        """
        if isinstance(truth, str):
            truth = truth.encode("utf-8")
        labels = numpy.zeros( (self.max_characters, self.n), dtype="float32" )
        for i, char in enumerate(truth):
            labels[i][ CIPHER.index(char)] = 1.0
        
        for i in range(len(truth), self.max_characters):
            labels[i][NULL] = 1.0
            
        return labels
    
    def decode(self, predictions):
        texts = []
        for p in predictions:
            #arr = p.numpy()
            arr = numpy.argmax(p, axis=1)
            text = bytes( CIPHER[i] for i in arr if i < self.n )
            texts.append(text)
        return texts
        
    def shape(self, img):
        shape = (*self.input_size, 3)
        dest = numpy.zeros(shape)
        height = min( ( img.shape[0], shape[0] ) )
        width = min( ( img.shape[1], shape[1] ) )
        dest[0:height, 0:width, :] = img[0:height, 0:width, :]
        
        return dest
    def loadModel(self, model_name):
        model = keras.models.load_model(model_name)
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=1e-5), 
            loss = keras.losses.CategoricalCrossentropy(),
         )
        return model

def createModel( model_name ):
    cr = Caligrapher()
    cr.encodeString("one")
    cr.encodeString(b"two")
    model = cr.buildModel()
    model.save(model_name)

def trainModel(data_folder, model_file, save_file=None):
        
        if save_file is None:
            if ".h5" in model_file:
                save_file = model_file.replace(".h5", "-trained.h5")
            else:
                save_file = model_file + "-trained"
        
        cr = Caligrapher()
        model = cr.loadModel(model_file)
        images_names, labels_names = data.get_file_names(data_folder)
        
        CHUNK=5000
        for j in range(100):
            for i in range(0, len(images_names), CHUNK):
                img_chunk = images_names[i:i+CHUNK]
                lbl_chunk = labels_names[i:i+CHUNK]
                boxes, letters = data.get_text_data(data_folder, img_chunk, lbl_chunk )
                tiles = numpy.array([ cr.shape(region) for region in boxes])
                lets = numpy.array([ cr.encodeString(text) for text in letters])
                
                model.fit(x = tiles, y = lets, batch_size=128, epochs = 20)

            y = model(tiles[:2])
            loser = model.loss(lets[:2], y).numpy()
            
            print( "loss of: ", loser)
            if math.isnan(loser):
                break
            print("grand epoch #", j)
            model.save(save_file)

def predictExamples():
    cr = Caligrapher()
    m = cr.loadModel("models/text-getter-trained")
    boxes, letters = data.get_text_data(
        "data/fake-simple", 
        ["img_1.jpg", "img_2.jpg"], 
        ["gt_img_1.txt", "gt_img_2.txt"]
    )
    tiles = numpy.array([ cr.shape(region) for region in boxes])
    lets = numpy.array([ cr.encodeString(text) for text in letters])
    
    p = m(tiles)
    guesses = cr.decode(p)
    original = cr.decode(lets)
    for i in range(len(letters)):
        print("->",guesses[i])
        print("__", original[i])


if __name__=="__main__":
    if sys.argv[1] == "c":
        createModel("models/text-getter")
    elif sys.argv[1] == "t":
        trainModel("data/fake-simple", "models/text-getter")
    elif  sys.argv[1] == "p":
        predictExamples()
