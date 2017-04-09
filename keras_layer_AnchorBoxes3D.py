import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np

class AnchorBoxes3D(Layer):
    '''
    Create an output tensor containing anchor boxes based on the input tensor and
    the passed arguments.

    A set of 3D anchor boxes of different is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the argument
    `anchor_lwhs`. The boxes are created for a bird's eye view input image and are
    parameterized by `(x0, x1, x2, x3, y0, y1, y2, y3, h)`, where pk = (xk, yk) represents
    the kth ground plane corner point of the 3D box and h is the box's height. All boxes
    are based on the ground plane. p0 and p1 are the top left and right corner points,
    p3 and p4 are the bottom left and right corner points.

    The logic implemented by this layer is identical to the logic in the module
    `ssd3Dbv_box_encode_decode_utils.py`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting box coordinates directly), one needs to know the anchor
    boxes in order to construct the prediction boxes from the offsets. If the model
    didn't contain this layer, one would always need to be able to generate the appropriate
    anchor box tensor externally and ad hoc for inference, which would be impossible for
    someone who only has the model itself. The reason why it is necessary to predict offsets
    to the anchor boxes rather than to predict box coordinates directly will be explained
    elsewhere.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 9)`.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_anchor_lwhs,
                 **kwargs):
        '''
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            this_anchor_lwhs (array): A 2D Numpy array of shape `(n, 3)` where the last axis contains
                `[length, width, height]` for each of the `n` box shapes for this classifier layer.
                Note that `n` is the number of boxes per cell, not the total number of boxes for the
                classifier layer.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1) or (next_scale > 1):
            raise ValueError("`this_scale` and `next_scale` must be in [0, 1], but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))
        self.img_height = img_height
        self.img_width = img_width
        self.this_anchor_lwhs = this_anchor_lwhs
        self.n_boxes = this_anchor_lwhs.shape[0] # Compute the number of boxes per cell
        super(AnchorBoxes3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes3D, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the input tensor.

        The logic implemented here is identical to the logic in the module `ssd3Dbv_box_encode_decode_utils.py`.

        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once for each classification conv layer during graph creation and is just being output
        along with the rest of the model output during runtime. Because of this, all logic is implemented
        as Numpy array operations and it sufficient to convert the resulting Numpy array into a Keras tensor
        at the very end before outputting it.
        '''
        # Compute box lengths, widths and heights as fractions of the shorter image side
        size = min(self.img_height, self.img_width)
        lwh = size * self.this_anchor_lwhs # 2D array of shape `(n_boxes, lwh values)`

        # We need the shape of the input tensor
        if K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = x._keras_shape

        # Compute the grid of box center points. They are identical for all lwh combinations
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = np.linspace(cell_width/2, self.img_width-cell_width/2, feature_map_size[1])
        cy = np.linspace(cell_height/2, self.img_height-cell_height/2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 6)`
        # where the last axis will contain `(cx, cy, cz, l, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], self.n_boxes, 6))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = lwh[:, 2] / 2 # Set cz - all boxes are placed on the ground plane, so cz is just half of the box's height
        boxes_tensor[:, :, :, 3] = lwh[:, 0] # Set l
        boxes_tensor[:, :, :, 4] = lwh[:, 1] # Set w
        boxes_tensor[:, :, :, 5] = lwh[:, 2] # Set h

        # Converts coordinates from (cx, cy, cz, l, w, h) to (x1, x2, x3, x4, y1, y2, y3, y4, h),
        # the 'corner_points' format - where (xk, yk) are the coordinates of pk, the kth point of the
        # box ground plane and h is the height of the box. Note that p1 and p2 are the top left and right
        # points of the ground plane and p3 and p4 are the bottom left and right points.
        boxes_tensor2 = np.zeros((feature_map_size[0], feature_map_size[1], self.n_boxes, 9))

        boxes_tensor2[:, :, :, [0,2]] = np.expand_dims(boxes_tensor[:, :, :, 0] - (boxes_tensor[:, :, :, 3] / 2), axis=-1) # cx - 0.5l == x1, x3
        boxes_tensor2[:, :, :, [1,3]] = np.expand_dims(boxes_tensor[:, :, :, 0] + (boxes_tensor[:, :, :, 3] / 2), axis=-1) # cx + 0.5l == x2, x4
        boxes_tensor2[:, :, :, [4,6]] = np.expand_dims(boxes_tensor[:, :, :, 1] - (boxes_tensor[:, :, :, 4] / 2), axis=-1) # cy - 0.5w == y1, y3
        boxes_tensor2[:, :, :, [5,7]] = np.expand_dims(boxes_tensor[:, :, :, 1] + (boxes_tensor[:, :, :, 4] / 2), axis=-1) # cy + 0.5w == y2, y4
        boxes_tensor2[:, :, :, 8] = boxes_tensor[:, :, :, 5] # h == h

        # Now prepend one dimension to `boxes_tensor2` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 9)`
        boxes_tensor2 = np.expand_dims(boxes_tensor2, axis=0)
        boxes_tensor2 = K.tile(K.constant(boxes_tensor2, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor2

    def compute_output_shape(self, input_shape):
        if K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 9)
