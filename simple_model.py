import cv2
from model import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow.keras import backend as K
from tqdm import tqdm

def main():
    args = parse_args()
    content_image = preprocess_image(args.content, size=(256, 256))
    style_image = preprocess_image(args.style, size=(content_image.shape[2], content_image.shape[1]))

    cv2.imwrite('output_images/content_base.png', deprocess_image(content_image))
    cv2.imwrite('output_images/style_base.png', deprocess_image(style_image))

    print('Loaded Images')
    print(content_image.shape)

    content_variable = tf.Variable(content_image, name='content-image')
    # K.get_session().run(tf.global_variables_initializer())
    # cv2.imwrite('content_variable.png', deprocess_image(K.get_session().run(content_variable)))
    # cv2.imwrite('content_base.png', deprocess_image(content_image))
    # exit()
    vgg = tf.keras.applications.vgg16.VGG16(input_shape=content_image.shape[1:],
                                            include_top=False, weights='imagenet')
    vgg.trainable = False

    feature_outputs = [vgg.get_layer(name).output for name in model_output_names]
    feature_model = keras.models.Model(vgg.input, feature_outputs)
    raw_style_features, target_content_features = get_intial_features(feature_model, content_image, style_image)

    full_model_input = kl.Input(tensor=content_variable)
    full_model_output = vgg(full_model_input)

    model_outputs = get_scope_output(K.get_session().graph, prefix='vgg16/', op_filter=model_output_names)

    print('Loaded Model')

    target_style_features = [gram_matrix(layer) for layer in raw_style_features]
    style_features = model_outputs[:len(style_layers)]
    content_features = model_outputs[len(style_layers):]

    loss = transfer_loss(style_features, content_features,
                         target_style_features, target_content_features, content_variable, style_image,
                         content_weight=1e3, style_weight=1e-2, total_variation_weight=1, colour_loss_weight=0)
    train_op = tf.train.AdamOptimizer(learning_rate=10).minimize(loss, var_list=[content_variable])

    sess = tf.Session()
    train(sess, content_variable, [train_op, loss])

if __name__ == '__main__':
    main()
