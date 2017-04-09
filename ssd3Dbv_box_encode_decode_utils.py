'''
Includes:
* Function to decode SSD3DBV model output
* Class to encode targets for SSD3DBV model training
'''

import numpy as np

def decode_y(y_pred):
    '''
    Convert SSD3DBV model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    Arguments:
        y_pred (array): The prediction output of the SSD3DBV model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 9 + 9)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 9 predicted coordinate offsets, 9 anchor box coordinates]`
            and the coordinate format is `(x0, x1, x2, x3, y0, y1, y2, y3, h)`. pk = (xk, yk) represents
            the kth ground plane corner point of the 3D box and h is the box's height.
            p0 and p1 are the corner points of the front side of the box.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 10)` where each row is a 3D box prediction for
        a non-background class for the respective image in the format `(class_id, x0, x1, x2, x3, y0, y1, y2, y3, h)`,
        where pk = (xk, yk) represents the kth ground plane corner point of the 3D box and h is the box's height.
        p0 and p1 are the corner points of the front side of the box.
    '''
    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:,:,-19:-9]) # Slice out the nine offset predictions plus one element where we'll write the class IDs in the next step
    y_pred_converted[:,:,0] = np.argmax(y_pred[:,:,:-18], axis=-1)

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    y_pred_converted[:,:,-1] = np.exp(y_pred_converted[:,:,-1]) # exp(ln(h(pred)/h(anchor))) == h(pred) / h(anchor)
    y_pred_converted[:,:,-1] *= y_pred[:,:,-1] # (h(pred) / h(anchor)) * h(anchor) == h(pred)
    y_pred_converted[:,:,-9:-1] *= np.linalg.norm(y_pred[:,:,[-9, -5]] - y_pred[:,:,[-6, -2]], axis=-1, keepdims=True) # ((pred - anchor) / diagonal(anchor)) * diagonal(anchor) = pred - anchor
    y_pred_converted[:,:,-9:-1] += y_pred[:,:,-9:-1] # (pred - anchor) + anchor = pred

    # 3: Decode our huge `(batch, #boxes, 10)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted: # For each image in the batch...
        y_pred_decoded.append(batch_item[np.nonzero(batch_item[:,0])]) # ...get all boxes that don't belong to the background class

    return y_pred_decoded

class SSD3DBVBoxEncoder:
    '''
    A ground truth label encoder for a SSD3DBV model, a FCN with SSD architecture
    to predict 3D bounding boxes for object detection in bird's eye view images
    where all objects are assumed to lie on the ground plane. The latter assumption
    implies a few significant simplifications for the network compared to the general
    case where the depth position of objects in the image can be arbitrary.

    The encoder transforms ground truth labels (3D bounding boxes and associated
    class labels) into the format required for training an SSD3DBV model.

    In the process of encoding ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an L2 box centroid distance threshold criterion.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 classifier_sizes,
                 anchor_lwhs,
                 pos_thresh,
                 neg_thresh):
        '''
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of classes including the background class.
            classifier_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional classifier layers.
            anchor_lwhs (array): A 3D Numpy array of shape `(m, n, 3)` where the last axis contains
                `[length, width, height]` as a fraction of the shorter image side for each
                of the `n` box shapes of each of the `m` classifier layers.
                Note that `n` is the number of boxes per cell, not the number of boxes per classifier layer.
            pos_thresh (float): The upper bound for the L2-distance between two box centroids in order
                to match a given ground truth box to a given anchor box. The centroids used are the centroids
                of the 2D box that is the bottom side of the
            neg_thresh (float): The lower bound for the L2-distance between a given anchor box and any
                ground truth box to be labeled a negative (i.e. background) box. If an anchor box is
                neither a positive, nor a negative box, it will be ignored during training.
        '''
        classifier_sizes = np.array(classifier_sizes)
        if len(classifier_sizes.shape) == 1:
            classifier_sizes = np.expand_dims(classifier_sizes, axis=0)

        anchor_lwhs = np.array(anchor_lwhs)
        if len(anchor_lwhs.shape) == 2:
            anchor_lwhs = np.expand_dims(anchor_lwhs, axis=0)

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.classifier_sizes = classifier_sizes
        self.anchor_lwhs = anchor_lwhs
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.n_boxes = anchor_lwhs.shape[1] # The number of boxes per cell

    def generate_anchor_boxes(self,
                              batch_size,
                              feature_map_size,
                              this_anchor_lwhs,
                              diagnostics=False):
        '''
        Compute an array of the spatial positions and sizes of the anchor boxes for one particular classification
        layer of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            batch_size (int): The batch size.
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            this_anchor_lwhs (array): A 2D numpy array of shape `(n_boxes, 3)`, where the last axis contains
                the length, width, height values (in this order) as fractions of the shorter image side,
                i.e. all three values are floats in [0,1].
            diagnostics (bool, optional): If true, two additional outputs will be returned.
                1) An array containing `(length, width, height)` values of the anchor boxes .
                2) A tuple `(cell_height, cell_width)` meaning how far apart the box centroids are placed
                   vertically and horizontally.
                This information is useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        Returns:
            A 3D Numpy array of shape `(batch, feature_map_height * feature_map_width * n_boxes, 6)` where the
            last dimension contains `(x0, x1, x2, x3, y0, y1, y2, y3, h)` for each anchor box in each cell
            of the feature map. `(xk, yk)` are the coordinates of `pk`, the kth point of the
            # box ground plane and `h` is the height of the box. Note that `p0` and `p1` are the top left and right
            # points of the ground plane and `p2` and `p3` are the bottom left and right points.
        '''
        # Compute box lengths, widths and heights as fractions of the shorter image side
        size = min(self.img_height, self.img_width)
        lwh = size * this_anchor_lwhs # 2D array of shape `(n_boxes, lwh values)`

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

        # Converts coordinates from (cx, cy, cz, l, w, h) to (x0, x1, x2, x3, y0, y1, y2, y3, h),
        # the 'corner_points' format - where (xk, yk) are the coordinates of pk, the kth point of the
        # box ground plane and h is the height of the box. Note that p0 and p1 are the top left and right
        # points of the ground plane and p2 and p3 are the bottom left and right points.
        boxes_tensor2 = np.zeros((feature_map_size[0], feature_map_size[1], self.n_boxes, 9))

        boxes_tensor2[:, :, :, [0,2]] = np.expand_dims(boxes_tensor[:, :, :, 0] - (boxes_tensor[:, :, :, 3] / 2), axis=-1) # cx - 0.5l == x1, x3
        boxes_tensor2[:, :, :, [1,3]] = np.expand_dims(boxes_tensor[:, :, :, 0] + (boxes_tensor[:, :, :, 3] / 2), axis=-1) # cx + 0.5l == x2, x4
        boxes_tensor2[:, :, :, [4,6]] = np.expand_dims(boxes_tensor[:, :, :, 1] - (boxes_tensor[:, :, :, 4] / 2), axis=-1) # cy - 0.5w == y1, y3
        boxes_tensor2[:, :, :, [5,7]] = np.expand_dims(boxes_tensor[:, :, :, 1] + (boxes_tensor[:, :, :, 4] / 2), axis=-1) # cy + 0.5w == y2, y4
        boxes_tensor2[:, :, :, 8] = boxes_tensor[:, :, :, 5] # h == h

        # Take cx and cy from `boxes_tensor`, we'll need them later
        anchor_centroids = boxes_tensor[:, :, :, :2]

        # Now prepend one dimension to `boxes_tensor2` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 9)`
        boxes_tensor2 = np.expand_dims(boxes_tensor2, axis=0)
        boxes_tensor2 = np.tile(boxes_tensor2, (batch_size, 1, 1, 1, 1))

        anchor_centroids = np.expand_dims(anchor_centroids, axis=0)
        anchor_centroids = np.tile(anchor_centroids, (batch_size, 1, 1, 1, 1))

        # Now reshape the 5D tensor above into a 3D tensor of shape
        # `(batch, feature_map_height * feature_map_width * n_boxes, 9)`. The resulting
        # order of the tensor content will be identical to the order obtained from the reshaping operation
        # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
        # use the same default index order, which is C-like index ordering)
        boxes_tensor2 = np.reshape(boxes_tensor2, (batch_size, -1, 9))

        anchor_centroids = np.reshape(anchor_centroids, (batch_size, -1, 2))

        if diagnostics:
            return boxes_tensor2, anchor_centroids, lwh, (int(cell_height), int(cell_width))
        else:
            return boxes_tensor2, anchor_centroids

    def generate_encode_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the conv net model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all classifier conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 18)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 18` because the model
            output contains not only the 9 predicted box coordinate offsets, but also the 9 coordinates for
            the anchor boxes.
        '''
        # 2: For each conv classifier layer get the tensors for
        #    the box coordinates of shape `(batch, n_boxes_total, 9)`
        boxes_tensor = []
        anchor_centroids = []
        if diagnostics:
            lwh_list = [] # List to hold the box widths and heights
            cell_sizes = [] # List to hold horizontal and vertical distances between any two boxes
            for i in range(self.anchor_lwhs.shape[0]):
                boxes, centroids, lwh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                 feature_map_size=self.classifier_sizes[i],
                                                                 this_anchor_lwhs=self.anchor_lwhs[i],
                                                                 diagnostics=True)
                boxes_tensor.append(boxes)
                anchor_centroids.append(centroids)
                lwh_list.append(lwh)
                cell_sizes.append(cells)
        else:
            for i in range(self.anchor_lwhs.shape[0]):
                boxes, centroids = self.generate_anchor_boxes(batch_size=batch_size,
                                                              feature_map_size=self.classifier_sizes[i],
                                                              this_anchor_lwhs=self.anchor_lwhs[i],
                                                              diagnostics=False)
                boxes_tensor.append(boxes)
                anchor_centroids.append(centroids)

        boxes_tensor = np.concatenate(boxes_tensor, axis=1)
        anchor_centroids = np.concatenate(anchor_centroids, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Concatenate the classes and boxes tensors to get our final template for y_encoded. We also need
        #    to append a dummy tensor of the shape of `boxes_tensor` so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this dummy tensor is irrelevant, it won't be
        #    used, so we'll just use `boxes_tensor` a second time.
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor), axis=2)

        if diagnostics:
            return y_encode_template, anchor_centroids, lwh_list, cell_sizes
        else:
            return y_encode_template, anchor_centroids

    def encode_y(self, ground_truth_labels):
        '''
        Convert ground truth bounding box data into a suitable format to train an SSD3DBV model.

        For each image in the batch, each ground truth bounding box belonging to that image will be compared against each
        anchor box in a template with respect to the L2-distances of the respective centroids of the 2D boxes that represent
        the bottom side of the full 3D boxes. If the L2-distance is smaller than
        or equal to `pos_thresh`, the boxes will be matched, meaning that the ground truth box coordinates and class
        will be written to the the specific position of the matched anchor box in the template.

        The class for all anchor boxes which have an L2-distance of greater than `neg_thresh` to all ground truth boxes
        will be set to the background class. Boxes that meet neither the positive nor the negative threshold criterion
        will be ignored, meaning their one-hot class encoding will be a vector of all zeros.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, x0, x1, x2, x3, y0, y1, y2, y3, h)`, where pk = (xk, yk) represents the kth ground plane
                corner point of the 3D box and h is the box's height. p0 and p1 are the corner points of the front
                side of the box. `class_id` must be an integer greater than 0 for all boxes as class_id 0 is reserved
                for the background class.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 9 + 9)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The 9 elements after the class vecotrs in
            the last axis are the box coordinates, and the last 9 elements are just dummy elements.
        '''

        # 1: Generate the template for y_encoded
        y_encode_template, anchor_centroids = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
        y_encoded = np.copy(y_encode_template) # We'll write the ground truth box data to this array

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

        class_vector = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(y_encode_template.shape[0]): # For each batch item...
            available_boxes = np.ones((y_encode_template.shape[1])) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
            negative_boxes = np.ones((y_encode_template.shape[1])) # 1 for all negative boxes, 0 otherwise
            for true_box in ground_truth_labels[i]: # For each ground truth box belonging to the current batch item...
                centroid = [[(true_box[-9] + true_box[-6]) / 2, (true_box[-5] + true_box[-2]) / 2]] # Compute the ground plane centroid of the grount truth box by taking the mean of two diagonal ground plane corner points
                similarities = np.linalg.norm(anchor_centroids[i, :, :] - centroid, axis=-1) # The L2 distances between the current ground truth box and all anchor boxes
                negative_boxes[similarities <= self.neg_thresh] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[available_and_thresh_met > self.pos_thresh] = 0 # Filter out anchor boxes which don't meet the iou threshold
                assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0: # If we have any matches
                    y_encoded[i,assign_indices,:-9] = np.concatenate((class_vector[true_box[0]], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                    available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    y_encoded[i,best_match_index,:-9] = np.concatenate((class_vector[true_box[0]], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1

        # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
        y_encoded[:,:,-18:-10] -= y_encode_template[:,:,-18:-10] # (gt - anchor) for all x-y-coordinates
        y_encoded[:,:,-18:-10] /= np.linalg.norm(y_encode_template[:,:,[-18, -14]] - y_encode_template[:,:,[-15, -11]], axis=-1, keepdims=True) # (gt - anchor) / diagonal(anchor)
        y_encoded[:,:,-10] /= y_encode_template[:,:,-10] # h(gt) / h(anchor)
        y_encoded[:,:,-10] = np.log(y_encoded[:,:,-10]) # ln(h(gt) / h(anchor)) (natural logarithm)

        return y_encoded
