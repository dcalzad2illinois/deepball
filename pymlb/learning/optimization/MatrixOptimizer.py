from typing import List, Dict
import numpy as np
import tensorflow as tf


class MatrixOptimizer:
    def __init__(self, variables: Dict[str, np.ndarray], inputs: List, iterations: int = 1000):
        # make sure there is no intersection in variable names
        if len(set(variables.keys()) & set(inputs)) > 0:
            raise ValueError("The variable and input names must be disjoint.")

        self.variables = variables
        self.inputs = inputs
        self.iterations = iterations

    def _create_loss(self, inputs):
        raise NotImplementedError

    def _projection(self, variables):
        return []

    def _optimize(self, **kwargs):
        # make sure that all input variables were specified
        for input_name in self.inputs:
            if input_name not in kwargs:
                raise ValueError("You must specify an input value for " + input_name + ".")
            if not isinstance(kwargs[input_name], np.ndarray):
                raise ValueError("The input value for " + input_name + " must be a numpy array.")

        # start up TensorFlow
        with tf.Session() as sess:
            # create the inputs
            inputs = {}
            for input_name in self.inputs:
                inputs[input_name] = tf.get_variable(input_name, initializer=tf.constant(kwargs[input_name], dtype=tf.float32), trainable=False)
            for variable_name, value in self.variables.items():
                inputs[variable_name] = tf.get_variable(variable_name, initializer=tf.constant(value, dtype=tf.float32))

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # create the loss function
            loss = self._create_loss(inputs)
            train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
            for i in range(self.iterations):
                sess.run(train_step)
                sess.run(self._projection(inputs))

            outputs = {}
            for variable_name in self.variables:
                outputs[variable_name] = sess.run(inputs[variable_name])
            return outputs

    def optimize(self, **kwargs):
        return self._optimize(**kwargs)
