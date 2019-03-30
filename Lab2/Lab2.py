
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "content_img/utahSkiLandscape.jpg" 
STYLE_IMG_PATH = "style_img/starryNight.jpg"
FINAL_IMG_FILE = 'output_img/starryNight_UtahSkiLandscape'


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3


# ============================= < Helper Fuctions > ============================= #
'''
Think I fixed this: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    if K.image_data_format() == 'channels_first':
        img = img.reshape((3, img_nrows, img_ncols))
        img = img.transpose((1, 2, 0))
    else:
        img = img.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# ============================= < Loss Function Builder Functions > ============================= #
# gen --> generated image
def styleLoss(style, gen):
    G = gramMatrix(gen)
    A = gramMatrix(style)
    return K.sum(K.square(G - A)) / (4. * 3^2 * (CONTENT_IMG_H * CONTENT_IMG_W)^2)

def contentLoss(content, gen):
    return K.sum(K.square(gramMatrix(gen) - gramMatrix(content)))

def totalLoss(x):
    a = K.square(x[:, :, :CONTENT_IMG_W - 1, :CONTENT_IMG_H - 1] - x[:, :, 1:, :CONTENT_IMG_H - 1])
    b = K.square(x[:, :, :CONTENT_IMG_W - 1, :CONTENT_IMG_H - 1] - x[:, :, :CONTENT_IMG_W - 1, 1:])
    return K.sum(K.pow(a + b, 1.25))
    # TODO: Understand this ^^^^^

# ============================= < Pipeline Functions > ============================= #

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def f_minLoss():
    # TODO: Implement this
    return 

def f_minGrads():
    # TODO: Implement this
    return

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    # Process and create the 3 images
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # Ok here we smash everything together into 1 tensor
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    # Give it to the model
    model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputTensor)
    # Gett the layers
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    
    # All this is loss 
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    # We do content loss here
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput) # I dunno if this is right --- OK I think it is
    # we do style loss here
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        layer = outputDict[styleLayerNames]
        styleOutput = layer[1, :, :, :]
        genOutput = layer[2, :, :, :]
        w = STYLE_WEIGHT / len(styleLayerNames)
        layerLoss = styleLoss(styleOutput, genOutput) #hmmmmmmmmmmmmmmmmm we are thinking, loading, processing
        loss += w * layerLoss # Fingers crossed
    # and here we do total loss
    loss += TOTAL_WEIGHT * totalLoss(genTensor)  # OK WE ARE MAKING PROGRESS
    
    # Setup gradients or use K.gradients() -- maybe I'm writing ok code
    gradients = K.gradients(loss, genTensor)

    outputs = K.function([genTensor], [loss].append(grads))

    # This loop is done
    print("   Beginning transfer.")
    imgData = tData.copy()
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        
        minEst, minVal, infoDict = fmin_l_bfgs_b(f_minLoss, tData.flatten(), fprime=f_minGrads, maxfun=20)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(imgData.copy())
        saveFile = FINAL_IMG_FILE + '_' + i + '.jpg'
        imsave(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")


# ============================= < Main > ============================= #
def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye :D ")

if __name__ == "__main__":
    main()
