import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation

from keras_layer_AnchorBoxes3D import AnchorBoxes3D

def build_model(image_size,
                n_classes,
                anchor_lwhs):
    '''
    Build a SSD3DBV Keras model, a FCN with SSD architecture to predict
    3D bounding boxes for object detection in bird's eye view images
    where all objects are assumed to lie on the ground plane. The latter assumption
    implies a few significant simplifications for the network compared to the general
    case where the depth position of objects in the image can be arbitrary.
    See references for a link to the SSD paper.

    The model consists of convolutional feature layers and one convolutional
    classifier layer that takes its input from the last feature layer.

    Note: Requires Keras v2.0 or later. Training currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of categories for classification including
            the background class (i.e. the number of positive classes +1 for
            the background calss).
        anchor_lwhs (array): A 3D Numpy array of shape `(m, n, 3)` where the last axis contains
            `[length, width, height]` for each of the `n` box shapes of each of the `m` classifier layers.
            Note that `n` is the number of boxes per cell, not the number of boxes per classifier layer.

    Returns:
        model: The Keras SSD3DBV model. It outputs predictions of shape `(batch_size, #boxes, 18)`,
            where the last axis contains `(...one-hot-encoded class labels..., x0, x1, x2, x3, y0, y1, y2, y3, h)`,
            where pk = (xk, yk) represents the kth ground plane corner point of the 3D box
            and h is the box's height. p0 and p1 are the corner points of the front side of the box.
            `#boxes` is the number of predicted boxes per image.
        classifier_sizes: A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional classifier. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_classifier_layers = 1 # The number of classifier conv layers in the network

    # Compute the number of boxes per cell
    anchor_lwhs = np.array(anchor_lwhs)
    if len(anchor_lwhs.shape) == 2:
        anchor_lwhs = np.expand_dims(anchor_lwhs, axis=0)
    n_boxes = anchor_lwhs.shape[1]

    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    # Design the actual network
    x = Input(shape=(img_height, img_width, img_channels))
    normed = Lambda(lambda z: z/127.5 - 1., # Convert input feature range to [-1,1]
                    output_shape=(img_height, img_width, img_channels),
                    name='lambda1')(x)


    conv1 = Convolution2D(32, (5, 5), name='conv1', strides=(1, 1), padding="same")(normed)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Convolution2D(48, (3, 3), name='conv2', strides=(1, 1), padding="same")(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Convolution2D(64, (3, 3), name='conv3', strides=(1, 1), padding="same")(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Convolution2D(64, (3, 3), name='conv4', strides=(1, 1), padding="same")(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Convolution2D(64, (3, 3), name='conv5', strides=(1, 1), padding="same")(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)

    # Build the convolutional classifier on top of conv layer 5
    # We build two classifiers: One for classes (classification), one for boxes (localization)
    # We precidt a class for each box, hence the classes classifiers have depth `n_boxes * n_classes`
    # We predict 9 box coordinates `(x1, x2, x3, x4, y1, y2, y3, y4, h)` for each box, hence the
    # boxes classifiers have depth `n_boxes * 9`
    # Output shape of classes: `(batch, height, width, n_boxes * n_classes)`
    classes5 = Convolution2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="valid", name='classes4')(conv5)
    # Output shape of boxes: `(batch, height, width, n_boxes * 9)`
    boxes5 = Convolution2D(n_boxes * 9, (3, 3), strides=(1, 1), padding="valid", name='boxes4')(conv5)
    # Generate the anchor boxes
    # Output shape of anchors: `(batch, height, width, n_boxes, 9)`
    anchors5 = AnchorBoxes3D(img_height, img_width, this_anchor_lwhs=anchor_lwhs[0], name='anchors5')(boxes5)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes in an isolated last axis to perform softmax on
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 9)`
    # We want the box coordinates in an isolated last axis to compute the smooth L1 loss
    boxes5_reshaped = Reshape((-1, 9), name='boxes5_reshape')(boxes5)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 9)`
    anchors5_reshaped = Reshape((-1, 9), name='anchors5_reshape')(anchors5)

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_final = Activation('softmax', name='classes_final')(classes5_reshaped)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 9 + 9)
    predictions = Concatenate(axis=2, name='predictions')([classes_final, boxes5_reshaped, anchors5_reshaped])

    model = Model(inputs=x, outputs=predictions)

    # Get the spatial dimensions (height, width) of the classifier conv layers, we need them to generate the default boxes
    # The spatial dimensions are the same for the classes and boxes classifiers
    classifier_sizes = np.array(classes5._keras_shape[1:3])

    return model, classifier_sizes
