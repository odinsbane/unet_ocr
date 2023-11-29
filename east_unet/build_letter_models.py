"""
This will make a bounding box model and detector model
for detecting bounding boxes of text.

The bounding box model shouldn't change, but the text
detector will change to just looking for one character.

"""
from east_unet import detection
from east_unet import model

detector = detection.Caligrapher((16, 16), 1)
dm = detector.buildModel()
dm.save("models/letter-detector")

builder = model.ModelBuilder()
bbm = builder.build()
bbm.save("models/letter-bb")
