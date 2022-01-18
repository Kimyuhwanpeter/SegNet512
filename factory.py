from classification_models.models_factory import ModelsFactory
import copy
import tensorflow as tf
import efficientnet.model as eff

class BackbonesFactory(ModelsFactory):
    _default_feature_layers = {

        # List of layers to take features from backbone in the following order:
        # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
        # resolution (Height x Width) than input image.

        # VGG
        'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
        'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2')
        }

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        for k in self._models_delete:
            del all_models[k]
        return all_models

    def get_backbone(self, name, *args, **kwargs):
        #model_fn, _ = self.get(name)
        if name == "vgg16":
            model_fn = tf.keras.applications.VGG16
        if name == "vgg19":
            model_fn = tf.keras.applications.VGG19

        # if you need other model then you can add like upon form

        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]

Backbones = BackbonesFactory()