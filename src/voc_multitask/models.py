"""Selected model builders from the final PASCAL VOC2012 experiments.

The builders mirror the architectures used for the locked final experiments. Training,
checkpoint selection, target encoding, losses, and evaluation remain in the notebook.
"""
from __future__ import annotations

import keras


def build_frozen_xception_classifier(
    input_shape=(180, 180, 3),
    n_classes: int = 20,
    seed: int = 42,
):
    """Build the selected frozen-Xception multi-label classifier.

    Inputs are expected in raw RGB ``[0, 255]`` scale. The model applies the
    same augmentation and Xception preprocessing used in the final experiment.
    """
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed + 801),
            keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 802),
            keras.layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="reflect",
                seed=seed + 803,
            ),
            keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
                seed=seed + 804,
            ),
            keras.layers.RandomContrast(0.10, seed=seed + 805),
        ],
        name="xception_augmentation",
    )

    inputs = keras.Input(shape=input_shape, name="image")
    x = augmentation(inputs)
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="xception_preprocessing")(x)

    base = keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        pooling=None,
        input_shape=input_shape,
        name="xception_conv_base",
    )
    base.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="xception_global_average_pooling")(x)

    regularizer = keras.regularizers.l2(1e-4)
    x = keras.layers.Dense(
        256,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="transfer_hidden_256",
    )(x)
    x = keras.layers.BatchNormalization(name="transfer_hidden_256_bn")(x)
    x = keras.layers.Activation("relu", name="transfer_hidden_256_relu")(x)
    x = keras.layers.Dropout(0.50, seed=seed + 812, name="transfer_dropout_050")(x)
    x = keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="transfer_bottleneck_64",
    )(x)
    x = keras.layers.Dropout(0.25, seed=seed + 813, name="transfer_dropout_025")(x)
    outputs = keras.layers.Dense(
        n_classes,
        activation="sigmoid",
        name="multilabel_probabilities",
    )(x)
    return keras.Model(inputs, outputs, name="voc2012_frozen_xception")


def build_segmentation_unet(input_shape=(128, 128, 3), dropout_rate: float = 0.30):
    """Build the selected three-level binary foreground/background U-Net."""

    def conv_block(x, filters, name, dropout=0.0):
        for i in (1, 2):
            x = keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                name=f"{name}_conv{i}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"{name}_bn{i}")(x)
            x = keras.layers.Activation("relu", name=f"{name}_relu{i}")(x)
        if dropout:
            x = keras.layers.SpatialDropout2D(dropout, name=f"{name}_dropout")(x)
        return x

    inputs = keras.Input(shape=input_shape, name="image")
    e1 = conv_block(inputs, 32, "encoder1")
    e2 = conv_block(keras.layers.MaxPooling2D(2)(e1), 64, "encoder2")
    e3 = conv_block(keras.layers.MaxPooling2D(2)(e2), 128, "encoder3")
    b = conv_block(keras.layers.MaxPooling2D(2)(e3), 256, "bottleneck", dropout_rate)

    d3 = keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")(b)
    d3 = conv_block(keras.layers.Concatenate()([d3, e3]), 128, "decoder3")
    d2 = keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    d2 = conv_block(keras.layers.Concatenate()([d2, e2]), 64, "decoder2")
    d1 = keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    d1 = conv_block(keras.layers.Concatenate()([d1, e1]), 32, "decoder1")
    outputs = keras.layers.Conv2D(
        1, 1, activation="sigmoid", name="foreground_probability"
    )(d1)
    return keras.Model(inputs, outputs, name="voc2012_segmentation_unet")


def build_yolo_style_detector(
    input_shape=(256, 256, 3),
    n_classes: int = 20,
    grid_size: int = 8,
    box_slots: int = 2,
    seed: int = 42,
):
    """Build the selected Xception-backed two-slot YOLO-style detector.

    Inputs are expected in RGB ``[0, 1]`` scale, matching the detection input
    pipeline. The returned network emits raw ``(grid, grid, slots, 5+C)``
    predictions. The custom multi-task loss and VOC-compatible decoding/NMS are
    implemented in the project notebook.
    """
    inputs = keras.Input(shape=input_shape, name="detection_image")
    x = keras.layers.RandomContrast(0.10, seed=seed, name="random_contrast")(inputs)
    x = keras.layers.GaussianNoise(0.015, seed=seed, name="gaussian_noise")(x)
    x = keras.layers.ReLU(max_value=1.0, name="clip_augmented_rgb")(x)
    x = keras.layers.Rescaling(2.0, offset=-1.0, name="xception_preprocessing")(x)

    base = keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False
    x = base(x, training=False)

    regularizer = keras.regularizers.l2(1e-5)
    x = keras.layers.SeparableConv2D(
        512,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        name="detection_head_sepconv_1",
    )(x)
    x = keras.layers.BatchNormalization(name="detection_head_bn_1")(x)
    x = keras.layers.Activation("relu", name="detection_head_relu_1")(x)
    x = keras.layers.SeparableConv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
        name="detection_head_sepconv_2",
    )(x)
    x = keras.layers.BatchNormalization(name="detection_head_bn_2")(x)
    x = keras.layers.Activation("relu", name="detection_head_relu_2")(x)
    x = keras.layers.Dropout(0.25, seed=seed, name="detection_head_dropout")(x)

    prediction_depth = 5 + n_classes
    raw = keras.layers.Conv2D(
        box_slots * prediction_depth,
        1,
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer="zeros",
        name="raw_detection_output",
    )(x)
    outputs = keras.layers.Reshape(
        (grid_size, grid_size, box_slots, prediction_depth),
        name="grid_slot_predictions",
    )(raw)
    return keras.Model(inputs, outputs, name="voc2012_improved_yolo")
