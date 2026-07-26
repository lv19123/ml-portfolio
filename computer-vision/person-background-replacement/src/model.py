"""Mobile U-Net architecture based on MobileNetV2 encoder."""

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2


def build_mobile_unet(
    input_shape=(256, 256, 3),
    num_classes=1,
    encoder_weights="imagenet",
    freeze_encoder=False,
):
    """Build Mobile U-Net for person segmentation.

    The architecture mirrors the training notebook: MobileNetV2 is used as an
    encoder, selected intermediate activations are used as skip connections,
    and the decoder upsamples with Conv2DTranspose blocks.
    """
    inputs = layers.Input(shape=input_shape)
    base_model = MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=encoder_weights,
    )
    base_model.trainable = not freeze_encoder

    skip_names = [
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]

    x = base_model.output

    for skip in reversed(skips):
        filters = int(skip.shape[-1])
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Dropout(0.2)(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, 1, activation=activation)(x)

    return Model(inputs=inputs, outputs=outputs)


Mobile_UNet = build_mobile_unet
