"""
 Given a bounding box and an image, how can we say what the text is.
"""
from tensorflow import keras
import tensorflow
import numpy

CIPHER=b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.\" "
NULL=len(CIPHER) - 1
import east_unet.data as data
import sys
import math, random, pathlib

class Caligrapher:
    def __init__(self, input_size=None, max_characters=None):
        self.input_size = input_size
        self.max_characters = max_characters
        self.n = len(CIPHER) 
    def buildModel(self, filters=8):
        inp = keras.layers.Input( (self.input_size[0], self.input_size[1], 3) )
        x = inp
        x = keras.layers.Conv2D(filters, (3, 3) )(x)
        x = keras.layers.Conv2D(filters, (3, 3) )(x)
        x = keras.layers.Conv2D(filters*2, (3, 3))(x)
        x = keras.layers.Conv2D(filters*2, (3, 3) )(x)
        x = keras.layers.Conv2D(filters*4, (3, 3) )(x)
        x = keras.layers.Conv2D(filters*4, (3, 3))(x)
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
        dest = numpy.zeros(shape, dtype=img.dtype)
        height = min( ( img.shape[0], shape[0] ) )
        width = min( ( img.shape[1], shape[1] ) )

        cy = 0
        if height < shape[0]:
            cy = (shape[0] - height)//2
        cx = 0
        #if width < shape[1]:
        #    cx = (shape[1] - width)//2
        dest[cy: cy + height, cx: cx + width, :] = img[0:height, 0:width, :]
        
        return dest
    def loadModel(self, model_name):
        model = keras.models.load_model(model_name)
        self.input_size = model.input.shape[1:3]
        self.max_characters = model.output.shape[1]
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=1e-5), 
            loss = keras.losses.MeanSquaredError(),
         )
        return model

def createModel( model_name ):
    cr = Caligrapher((16, 96))
    cr.encodeString("one")
    cr.encodeString(b"two")
    model = cr.buildModel()
    model.save(model_name)

def getErrorByLetter(v_letters, v_pred):
    dp = (v_letters - v_pred)
    dp = dp*dp
    err = tensorflow.reduce_sum(dp, axis=(0, 1))
    tot = tensorflow.reduce_sum(v_letters, axis=(0, 1))
    err = err/tot
    return err
        
def getErrorByPosition(v_letters, v_pred):
    dp = (v_letters - v_pred)
    dp = dp*dp
    err = tensorflow.reduce_sum(dp, axis=(0, 2))
    tot = tensorflow.reduce_sum(v_letters, axis=(0, 2))
    err = err/tot
    return err

def trainModel(data_folder, model_file, save_file=None):
        if not isinstance(model_file, pathlib.Path):
            model_file = pathlib.Path(model_file)
            print("converting")
        if save_file is None:
            if ".h5" in model_file.name:
                save_file = pathlib.Path(model_file.parent, model_file.name.replace(".h5", "-trained.h5"))
            else:
                save_file = pathlib.Path(model_file.parent, model_file.name + "-trained")
        
        cr = Caligrapher()
        model = cr.loadModel(model_file)
        images_names, labels_names = data.getFileNames(data_folder)

        every = [
                    ( pathlib.Path(data_folder, i), pathlib.Path(data_folder, l) )
                    for i,l in zip(images_names, labels_names)
                    ]
        #every = every[:1000]
        n_validate = 500
        train = every[0:-n_validate]
        random.shuffle(train)

        valid = every[len(train):]
        images_names = [ il[0] for il in train ]
        labels_names = [ il[1] for il in train ]

        CHUNK=50000

        val_names = [il[0] for il in valid]
        val_labels = [il[1] for il in valid]

        vboxes, vletters = data.getTextData(val_names, val_labels)
        v_tiles = numpy.array([ cr.shape(region) for region in vboxes])
        v_letters = numpy.array([ cr.encodeString(text) for text in vletters])

        boxes, letters = data.getTextData(images_names, labels_names )

        with open("caligrapher-%s-loss.txt"%model_file.name, 'w') as logger:
            logger.write("#epoch\tchunk\tloss\n")
            v_pred = model(v_tiles)
            loser = model.loss(v_letters, v_pred).numpy()
            by_letter = open("letter-loss.txt", 'w')
            
            err = getErrorByPosition(v_letters, v_pred)
            for v in err.numpy():
                by_letter.write("%s\t"%v)
            by_letter.write("\n")
            by_letter.flush()
            
            best = loser
            print( "starting loss of: ", loser)
            logger.write("%s\t%s\t%s\n"%(-1, -1, loser))
            logger.flush()

            for j in range(100):
                for i in range(0, len(boxes), CHUNK):
                    tiles = numpy.array([ cr.shape(region) for region in boxes[i:i+CHUNK]])
                    lets = numpy.array([ cr.encodeString(text) for text in letters[i:i+CHUNK]])
                    model.fit(x = tiles, y = lets, batch_size=256, epochs =10)
                    v_pred = model(v_tiles)
                    loser = model.loss(v_letters, v_pred).numpy()
                    
                    err = getErrorByPosition(v_letters, v_pred)
                    for v in err.numpy():
                        by_letter.write("%s\t"%v)
                    by_letter.write("\n")
                    by_letter.flush()

                    print( "loss of: ", loser)
                    logger.write("%s\t%s\t%s\n"%(j, i, loser))
                    logger.flush()
                    if math.isnan(loser):
                        break
                if loser < best:
                    best = loser
                    model.save( pathlib.Path( model_file.parent, "%s-best"%(model_file.name) ) )
                print("grand epoch #", j)
                model.save(save_file)
def validate(data_file, model):
    gt = getGTFile(data_file)
    boxes, letters = data.getTextData([data_file], [gt])
    cr = Caligrapher()
    cr.loadModel(model)

    from matplotlib import pyplot

    pyplot.imshow( data.readImage(data_file) )
    pyplot.show()

    for box, lets in zip(boxes, letters):
        print(lets)
        pyplot.imshow( cr.shape(box) )
        pyplot.show()

def getGTFile(img_file):
    p = pathlib.Path(img_file)
    i = p.name.rfind(".")
    return pathlib.Path(p.parent, "gt_%s.txt"%(p.name[:i]))

def predictExamples(model, files):
    cr = Caligrapher()
    m = cr.loadModel(model)

    files = [pathlib.Path(f) for f in files]
    labels = [ getGTFile(f) for f in files]

    boxes, letters = data.getTextData(files, labels)
    tiles = numpy.array([ cr.shape(region) for region in boxes])
    lets = numpy.array([ cr.encodeString(text) for text in letters])
    
    p = m(tiles)
    guesses = cr.decode(p)
    original = cr.decode(lets)
    for i in range(len(letters)):
        print("->",guesses[i])
        print("__", original[i])


if __name__=="__main__":
    print("usage: ")
    if sys.argv[1] not in "ctpv":
        print("detection [ctp] *args")
        print("c: create; t: train; p: predict; v: validate")
    if sys.argv[1] == "c":
        createModel(sys.argv[2])
    elif sys.argv[1] == "t":
        trainModel(sys.argv[3], sys.argv[2])
    elif  sys.argv[1] == "p":
        predictExamples(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "v":
        validate(sys.argv[3], sys.argv[2])
