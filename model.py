import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow.keras import backend as K
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.ops import array_ops, variables
from tqdm import tqdm

# Tensors move values between ops and ops do actions
# For example an assign op will give the rest of the graph initial values
# to all global init is, is a list of assignment operators. Once those variables have
# values the assignments will never be run again because the graph is lazy.

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def style_loss(style, gram_target):
    # if len(style.shape) != 4 or len(gram_target.shape) != 4:
    #     raise ValueError('input to style loss must be 4-dimensional.')

    height, width, channels = style.shape[1:]
    gram_style = gram_matrix(style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def total_variation_loss(images):
    # if len(images.shape) != 4:
    #     raise ValueError('input to variation loss must be 4-dimensional.')

    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

    # Only sum for the last 3 axis.
    # This results in a 1-D tensor with the total variation for each image.
    sum_axis = [1, 2, 3]
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (tf.reduce_mean(tf.abs(pixel_dif1), axis=sum_axis) +
               tf.reduce_mean(tf.abs(pixel_dif2), axis=sum_axis))

    return tot_var

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
model_output_names = style_layers + content_layers

def continous_gen(X, Y, batch_size=None):
    idx = 0
    while True:
        if batch_size is None:
            yield X, Y
        else:
            start = idx*batch_size % len(self.X)
            end = idx*(batch_size+1) % len(self.X)
            idx += 1
            if end < start:
                start = start - len(self.X)

            yield self.X[start:end], self.Y[start:end]

def transfer_loss(style_features, content_features, target_style, target_content, image_variable,
                  content_weight=1e3, style_weight=1e-2, total_variation_weight=1):
    weight_per_style_layer = 1 / len(style_features)
    weight_per_content_layer = 1 / len(content_features)

    style_score = 0
    for feature, target in zip(style_features, target_style):
        style_score += weight_per_style_layer * style_loss(feature, target)

    content_score = 0
    for feature, target in zip(content_features, target_content):
        content_score += weight_per_content_layer * content_loss(feature, target)

    style_score *= style_weight
    content_score *= content_weight
    total_variation_score = total_variation_weight * tf.reduce_mean(total_variation_loss(image_variable))

    return style_score + content_score + total_variation_score

def get_intial_features(model, content_image, style_image):
    stacked_images = np.concatenate([style_image, content_image], axis=0)

    with K.get_session().as_default():
        output = K.get_session().run(model.output, feed_dict={model.input: stacked_images})
    target_style_features = [style_layer[0] for style_layer in output[:len(style_layers)]]
    target_content_features = [content_layer[1] for content_layer in output[len(style_layers):]]

    return target_style_features, target_content_features

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return cv2.hconcat(x)

def get_scope_output(graph, prefix=None, op_filter=None):
    operations = graph.get_operations()
    nodes =  []
    for op in operations:
        if prefix is not None:
            op_name = prefix.join(op.name.split(prefix)[1:])
        else:
            op_name = op.name

        for op_input in op.inputs:
            if prefix is not None:
                op_input_name = prefix.join(op_input.name.split(prefix)[1:])
            else:
                op_input_name = op_input.name

            if op_input_name.split('/')[0] != op_name.split('/')[0] and len(op_name.split('/')) > 1:
                nodes.append(op_input)

    node_dict = {}
    for node in nodes:
        if prefix is not None:
            node_name = prefix.join(node.name.split(prefix)[1:])
        else:
            node_name = node.name
        node_name = node_name.split('/')[0]
        if node_name not in node_dict:
            node_dict[node_name] = [node]
        else:
            node_dict[node_name].append(node)

    if op_filter is not None:
        model_outputs = [tensor for op in op_filter for tensor in node_dict[op]]
    else:
        model_outputs = [tensor for v in node_dict.values() for tensor in v]

    return model_outputs

def get_input_ops(graph):
    operations = graph.get_operations()
    graph_inputs = [op for op in operations if len(op.inputs) == 0]
    return graph_inputs

def get_output_tensors(graph):
    operations = graph.get_operations()

    output_dict = {}
    input_tensors = set()
    output_tensors = set()
    for op in operations:
        input_tensors.update(set([op_input.name for op_input in op.inputs if isinstance(op_input, tf.Tensor) or isinstance(op_input, tf.Variable)]))
        output_tensors.update(set([op_output.name for op_output in op.outputs if isinstance(op_output, tf.Tensor) or isinstance(op_output, tf.Variable)]))
        output_dict.update({op_output.name: op_output for op_output in op.outputs})

    return [output_dict[output] for output in output_tensors - input_tensors]

class FakeVariable(variables.RefVariable):
    def __init__(self, variable_op, initial_value, assignment_op, collections=None,
                 caching_device=None, constraint=None, trainable=True):
        if collections is None:
            collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if not isinstance(collections, (list, tuple, set)):
            raise ValueError("collections argument to Variable constructor must be a list, tuple, "
                             "or set. Got %s of type %s" % (collections, type(collections)))
        self._graph_key = tf.get_default_graph()._graph_key # pylint: disable=protected-access

        if isinstance(initial_value, checkpointable.CheckpointInitialValue):
            self._maybe_initialize_checkpointable()
            self._update_uid = initial_value.checkpoint_position.restore_uid
            initial_value = initial_value.wrapped_value

        if constraint is not None and not callable(constraint):
            raise ValueError("The `constraint` argument must be a callable.")

        self._trainable = trainable
        if trainable and tf.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [tf.GraphKeys.TRAINABLE_VARIABLES]

        self._initial_value = initial_value
        shape = self._initial_value.shape
        self._variable_op = variable_op
        self._variable = self._variable_op.outputs[0]
        self._initializer_op = assignment_op

        if caching_device is not None:
            with tf.device(caching_device):
                self._snapshot = array_ops.identity(self._variable, name="read")
        else:
            with tf.colocate_with(self._variable_op):
                self._snapshot = array_ops.identity(self._variable, name="read")
        tf.add_to_collections(collections, self)

        self._caching_device = caching_device
        self._save_slice_info = None
        self._constraint = constraint

    @property
    def op(self):
        """The `Operation` of this variable."""
        return self._variable_op

def train(sess, content_variable, to_pull):
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        cv2.imwrite('content_variable.png', deprocess_image(content_variable.eval()))
        for i in tqdm(range(1000)):
            _ = sess.run(to_pull)
            if i % 10 == 0:
                cv2.imwrite('content_variable_{}.png'.format((i % 100) // 10),
                            deprocess_image(content_variable.eval()))

        cv2.imwrite('content_variable.png', deprocess_image(content_variable.eval()))

def main():
    style_image = preprocess_image('style_image.png')
    content_image = preprocess_image('content_image.png')
    style_image = np.expand_dims(cv2.resize(style_image[0], content_image.shape[1:3]), axis=0)

    content_variable = tf.Variable(content_image, name='content-image')
    vgg = tf.keras.applications.vgg19.VGG19(input_shape=content_image.shape[1:],
                                            include_top=False, weights='imagenet')
    vgg.trainable = False

    feature_outputs = [vgg.get_layer(name).output for name in model_output_names]
    feature_model = keras.models.Model(vgg.input, feature_outputs)
    raw_style_features, target_content_features = get_intial_features(feature_model, content_image, style_image)

    full_model_input = kl.Input(tensor=content_variable)
    full_model_output = vgg(full_model_input)
    init = tf.global_variables_initializer()
    keras_graph = K.get_session().graph
    # keras_graph_inputs = [content_variable,]
    keras_graph_outputs = [full_model_output,]
    keras_graph_inputs = [op for op in get_input_ops(keras_graph) if op.type not in ['NoOp', 'Placeholder']] + [init]
    # keras_graph_outputs = get_output_tensors(keras_graph)

    transfer_graph = tf.Graph()
    sess = tf.Session(graph=transfer_graph)
    with transfer_graph.as_default():
        graph_interface = tf.import_graph_def(tf.graph_util.extract_sub_graph(keras_graph.as_graph_def(),
                                              [i.name.split(':')[0] for i in keras_graph_inputs + keras_graph_outputs]),
                                              name='transfer')
                                              # return_elements=[full_model_output.name.split(':')[0],
                                              #                  content_variable.name.split(':')[0]])
        graph_inputs = get_input_ops(transfer_graph)
        graph_outputs = get_output_tensors(transfer_graph)

        model_outputs = get_scope_output(transfer_graph, prefix='transfer/vgg19/', op_filter=model_output_names)

        sess.run(transfer_graph.get_operation_by_name('transfer/init_1'))

    # print(list(transfer_graph.as_graph_def().node)[0])

    style_features = model_outputs[:len(style_layers)]
    content_features = model_outputs[len(style_layers):]

    with transfer_graph.as_default():
        target_style_features = [gram_matrix(layer) for layer in raw_style_features]

        fake_input = FakeVariable(transfer_graph.get_operation_by_name('transfer/content-image'),
                                  transfer_graph.get_operation_by_name('transfer/content-image/initial_value').outputs[0],
                                  transfer_graph.get_operation_by_name('transfer/content-image/Assign'))

        loss = transfer_loss(style_features, content_features,
                             target_style_features, target_content_features, fake_input)
        train_op = tf.train.AdamOptimizer(learning_rate=10).minimize(loss, var_list=[fake_input])

        train(sess, transfer_graph.get_operation_by_name('transfer/content-image').outputs[0], [train_op, loss])

if __name__ == '__main__':
    main()
