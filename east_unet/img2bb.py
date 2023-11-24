

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
    """
        produces two arrays with the coordinates as values.

        x - [[ 0, 1, 2, 3, ...], [0, 1, 2, 3, ...], ...]
        y - [[0, 0, 0, ...], [1, 1, 1, ...], ...]
    """
    x = numpy.tile( numpy.arange( shape[1] ), (shape[0], 1) )
    y = numpy.zeros( shape[0:2] )
    for i in range(shape[0]):
        y[i, : ] = i
    return x, y

def relativeToAbsolute(distances):
    """
        Takes an array/tensor with 4 channels that represent the distances
        to the edges of the rectangular bounding box.

        Distances are derived from the bounding box lines going cw.
        x0,y0           x1, y1
        x3,y3           x2, y2

        The first line is x0, y0 to x1, y1 which gives us ymin
        The second line is x1, y1 to x2, y2 which gives us xmax
        The third line is x2, y2 to x3, y3 which gives us ymax
        The fourth line is x3, y3 to x0, y0 which gives us xmin

        return [ymin, xmin, ymax, xmax] the bb for that tensorflow uses.
    """
    x, y = getXYGrid( distances.shape )
    mask = ( numpy.sum(distances, axis=2) > 0 )*1

    y1 = (y*mask - distances[:, :, 0])
    x1 = (x*mask - distances[:, :, 3])
    y2 = (y*mask + distances[:, :, 2])
    x2 = (x*mask + distances[:, :, 1])

    distances[:, :, 0] = y1
    distances[:, :, 1] = x1
    distances[:, :, 2] = y2
    distances[:, :, 3] = x2

    return distances

def relImageToList( distances ):
    boxes = relativeToAbsolute(distances)
    bbs = tensorflow.reshape(boxes, (x*y, 4))
    weights = tensorflow.reshape(scores, (x*y, ))
    selected_indices = tensorflow.image.non_max_suppression(bbs, weights, x*y)
    selected_boxes = tensorflow.gather(bbs, selected_indices)
    selected_scores = tensorflow.gather(weights, selected_indices)

    return selected_boxes, selected_scores

def getGTFile(img_file):
    i = img_file.name.rfind(".")
    return pathlib.Path(img_file.parent, "gt_%s.txt"%(img_file.name[:i]))

def nms(distances, scores):
    x = distances.shape[1]
    y = distances.shape[0]
    bbs = tensorflow.reshape(distances, (x*y, 4))
    weights = tensorflow.reshape(scores, (x*y, ))
    selected_indices = tensorflow.image.non_max_suppression(bbs, weights, x*y)
    selected_boxes = tensorflow.gather(bbs, selected_indices)
    selected_scores = tensorflow.gather(weights, selected_indices)
    return selected_boxes, selected_scores

def bbImageAndLabel(image, rectangles):
    area = 0
    perimeter = 0
    for rectangle in rectangles:
        print(rectangle)
        area += (rectangle.maxy - rectangle.miny)*(rectangle.maxx - rectangle.minx)
        perimeter += 2*( (rectangle.maxy - rectangle.miny) + (rectangle.maxx - rectangle.minx) )
    print(area, perimeter)
    labelled = data.labelImage(image, rectangles)
    distances = relativeToAbsolute(numpy.array(labelled[:, :, 0:4], dtype="float32"))
    scores = numpy.array(labelled[:, :, 4], dtype="float32")
    return nms(distances, scores)

def bbImagePredict(model, image):
    op = model.predict(numpy.array([image]))
    b2 = relativeToAbsolute(op["boxes"][0])
    s2 = op["score"][0]
    bbs = tensorflow.reshape(b2, (x*y, 4))
    weights = tensorflow.reshape(s2, (x*y, ))
    selected_indices = tensorflow.image.non_max_suppression(bbs, weights, x*y)
    selected_boxes = tensorflow.gather(bbs, selected_indices)
    selected_scores = tensorflow.gather(weights, selected_indices)
    return selected_boxes, selected_scores

if __name__=="__main__":
    if sys.argv[1] not in ["l","g"]:
        print("usage: img2bb [lg] *options")
    if sys.argv[1] == "l":
        imgf = pathlib.Path(sys.argv[2])
        if len(sys.argv) > 3:
            lblf = pathlib.Path(sys.argv[3])
        else:
            lblf = getGTFile(imgf)

        if not lblf.exists():
            print("cannot find label file exiting.")
            print("\timg2bb l img.png [lbl.txt]")
            sys.exit(-1)
        image, rectangles = data.loadImageLabels(imgf, lblf)
        selected_boxes, selected_scores = bbImageAndLabel(image, rectangles)
    if sys.argv[1] == "g":
        if len(sys.argv) < 3:
            print("usage: img2bb g model image")
            sys.exit(0)
        image = data.readImage(pathlib.Path(sys.argv[3]))
        model = keras.models.load_model(sys.argv[2])

        selected_boxes, selected_scores = bbImagePredict(model, image)



    from matplotlib import pyplot


    good = []
    rectangles = []
    height = image.shape[0]
    width = image.shape[1]
    scales = numpy.array( [height, width, height, width] )
    for box, score in zip( selected_boxes, selected_scores):
        print(box, score)
        if( score > 0.5 ):
            rectangles.append(data.axisAlignedRectangle( box[0], box[1], box[2], box[3]))
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
