
import sys
from tensorflow import keras
import data
import numpy

import matplotlib.pyplot as pyplot

if __name__=="__main__":
    model = keras.models.load_model(sys.argv[1])
    img = data.read_image(sys.argv[2])
    print(model.outputs[0].shape)
    tiles = data.tile_img(img, (240, 320, 3))
    #for tile in tiles:
        #pyplot.imshow(tile[:,:,0])
        #pyplot.show()
        
    y = model(numpy.array(tiles))
    for i in range(len(tiles)):
        figure = pyplot.figure(i+1)
        zz = y[i]
        c = 1
        figure.add_subplot(2, 2, c);
        pyplot.imshow(zz[:, :, 0])
        c += 1
        figure.add_subplot(2, 2, c);
        pyplot.imshow(zz[:, :, 1])
        c += 1
        figure.add_subplot(2, 2, c);
        pyplot.imshow(zz[:, :, 2])
        c += 1
        figure.add_subplot(2, 2, c);
        pyplot.imshow(tiles[i][:, :, 0])
        pyplot.show()
