import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


def feature_extractor(inputs):
    feature_extractor = ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )(inputs)
    return feature_extractor


def main():
    # Instantiate a base miodel with pre-trained weights.
    base_model = ResNet50(
        include_top=True,
        weights="imagenet",
    )
    base_model.trainable = False

    layer_output = base_model.get_layer("avg_pool").output
    intermediate_model = tf.keras.models.Model(
        inputs=base_model.input, outputs=layer_output
    )

    print(base_model.summary())
    img_path = "images/doggo.jpeg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = intermediate_model.predict(x)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    print(preds.shape)


if __name__ == "__main__":
    main()
