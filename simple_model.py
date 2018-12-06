import cv2
from model import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow.keras import backend as K
from tqdm import tqdm

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

    model_outputs = get_scope_output(K.get_session().graph, prefix='vgg19/', op_filter=model_output_names)

    target_style_features = [gram_matrix(layer) for layer in raw_style_features]
    style_features = model_outputs[:len(style_layers)]
    content_features = model_outputs[len(style_layers):]

    loss = transfer_loss(style_features, content_features,
                         target_style_features, target_content_features, content_variable)
    train_op = tf.train.AdamOptimizer(learning_rate=10).minimize(loss, var_list=[content_variable])

    sess = tf.Session()
    train(sess, content_variable, [train_op, loss])

if __name__ == '__main__':
    main()
