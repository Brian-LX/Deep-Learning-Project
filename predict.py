import argparse
import cv2
import numpy as np
import imutils
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont
NORM_SIZE = 32

sign = { "0": "bacterial_leaf_blight", "1": "brown_spot", "2": "healthy","3": "leaf_blast","4": "leaf_scald", "5": "narrow_brown_spot"}

def haze_removal(image, windowSize=24, w0=0.6, t0=0.1):

    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    result = Image.fromarray(J)

    return result
def addText(img, text, i):
    # Determine whether the picture is in ndarray format and convert it to RGB
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    # The parameters are font, font size, and coding
    fontStyle = ImageFont.truetype("simsun.ttc", 20, encoding="utf-8")
    # The parameters are position, text, color, and font
    draw.text((10, (i * 50) + 23), text, (255, 255, 255), font=fontStyle)

    # Go back to BGR image, ndarray format
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def predict(path):
    print("loading model...")
    model = load_model("cnn.model")

    print("loading image...")

    # dehaze
    image = np.array(Image.open(path))
    imageSize = image.shape
    result = haze_removal(image)
    image = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

    images = cv2.imread(path)
    orig = images.copy()

    # preprocessing
    image = cv2.resize(image, (NORM_SIZE, NORM_SIZE))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    #Predict
    result = model.predict(image)[0]
    proba = np.max(result)
    # label = str(np.where(result==proba)[0])
    label = str(np.where(result == proba)[0][0])

    # labels = "{}: {:.2f}%".format(sign[label], proba*100)
    labels = sign[label]
    print(sign[label])

    origs = orig[:,:,[2,1,0]]
    output = imutils.resize(origs, width=400)
    imgshow = addText(output, labels, 0)
    # cv2.putText(output, labels, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.imshow("Output", imgshow)
    # cv2.waitKey(0)
    return imgshow, labels

if __name__ == "__main__":
    args ="training_1/1/brown_spot (1).JPG"
    predict(args)

