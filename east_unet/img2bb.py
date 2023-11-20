

from tensorflow import keras
import east_unet.data as data
import tensorflow.image
import tensorflow
import numpy
import sys
import pathlib

y = 240
x = 320

def getXYGrid( shape ):
    x = numpy.tile( numpy.arange( shape[1] ), (shape[0], 1) )
    y = numpy.zeros( shape[0:2] )
    for i in range(shape[0]):
        y[i, : ] = i
    return x, y

def relativeToAbsolute(distances):

    x, y = getXYGrid( distances.shape )
    mask = ( numpy.sum(distances, axis=2) > 0 )*1

    print("middle: ", distances[32, 30, :], y[32, 30], x[32, 30])
    print("left: ", distances[32, 25, :], y[32, 30], x[32, 25])
    print("right: ", distances[32, 35, :], y[32, 35], x[32, 35])
    print("above: ", distances[37, 30, :], y[37, 30], x[37, 30])
    print("below: ", distances[27, 30, :], y[27, 30], x[27, 30])

    print(numpy.sum(mask))
    y1 = (y*mask - distances[:, :, 0])
    x1 = (x*mask - distances[:, :, 3])
    y2 = (y*mask + distances[:, :, 2])
    x2 = (x*mask + distances[:, :, 1])

    distances[:, :, 0] = y1
    distances[:, :, 1] = x1
    distances[:, :, 2] = y2
    distances[:, :, 3] = x2
    return distances

print("usage: img2bb img lbl")

imgf = pathlib.Path(sys.argv[1])
lblf = pathlib.Path(sys.argv[2])

image, rectangles = data.loadImageLabels(imgf, lblf)

area = 0
perimeter = 0
for rectangle in rectangles:
    print(rectangle)
    area += (rectangle.maxy - rectangle.miny)*(rectangle.maxx - rectangle.minx)
    perimeter += 2*( (rectangle.maxy - rectangle.miny) + (rectangle.maxx - rectangle.minx) )

print(area, perimeter)

labelled = data.labelImage(image, rectangles)

distances = numpy.array(labelled[:, :, 0:4], dtype="float32")

scores = numpy.array(labelled[:, :, 4], dtype="float32")

boxes = relativeToAbsolute(distances)

bbs = tensorflow.reshape(boxes, (x*y, 4))

weights = tensorflow.reshape(scores, (x*y, ))

selected_indices = tensorflow.image.non_max_suppression(bbs, weights, x*y)

selected_boxes = tensorflow.gather(bbs, selected_indices)


from matplotlib import pyplot


for box in selected_boxes:
    print(box)
    if all(b == 0 for b in box):
        break
height = image.shape[0]
width = image.shape[1]
scales = numpy.array( [height, width, height, width] )
if len(sys.argv) > 3:
    model = keras.models.load_model(sys.argv[3])
    op = model.predict(numpy.array([image]))
    b2 = relativeToAbsolute(op["boxes"][0])
    #b2 = op["boxes"][0]
    s2 = op["score"][0]
    bbs = tensorflow.reshape(b2, (x*y, 4))
    weights = tensorflow.reshape(s2, (x*y, ))
    selected_indices = tensorflow.image.non_max_suppression(bbs, weights, x*y)
    selected_boxes = tensorflow.gather(bbs, selected_indices)
    selected_scores = tensorflow.gather(weights, selected_indices)
    good = []
    for box, score in zip( selected_boxes, selected_scores):
        print(box, score)
        if( score > 0.5 ):
            box2 = box/scales
            good.append(box2)
        else:
            break
    boxes = numpy.array([good])
    colors = numpy.array( [[0.0, 0.0, 1.0]]*len(good) )
    print(image.shape)
    rmx = 255 #numpy.max(image[:,:,0])
    gmx = 255 #numpy.max(image[:,:,1])
    bmx = 255 #numpy.max(image[:,:,2])
    fimg = numpy.array([image], dtype="float32")
    fimg[0,:,:,0] = fimg[0,:,:,0]/rmx
    fimg[0,:,:,1] = fimg[0,:,:,1]/gmx
    fimg[0,:,:,2] = fimg[0,:,:,2]/bmx

    drawn = tensorflow.image.draw_bounding_boxes(fimg, boxes, colors)
    pyplot.imshow(drawn[0])
    pyplot.show()
