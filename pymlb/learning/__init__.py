import os

import keras.models

# import sub-packages
from .aggregators import __init__
from .layers import __init__
from .optimization import __init__

from .ModelCheckpointParallel import ModelCheckpointParallel
from .Model import Model
from .AggregateModel import AggregateModel
from .DeepModel import DeepModel
from .RandomForestContinuousModel import RandomForestContinuousModel
from .MagicVectorDecoderModel import MagicVectorDecoderModel

# import the utilities which rely on the classes imported above
from .mlb import __init__
from .utilities import __init__

# from http://zachmoshe.com/2017/04/03/pickling-keras-models.html
def make_keras_picklable():
    def __getstate__(self):
        name = "tempmodel"
        keras.models.save_model(self, name, overwrite=True)
        with open(name, "rb") as fd:
            model_str = fd.read()
        os.remove(name)
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        name = "tempmodel"
        with open(name, "wb") as fd:
            fd.write(state['model_str'])
        model = keras.models.load_model(name, custom_objects=DeepModel.get_custom_objects())
        os.remove(name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


make_keras_picklable()
