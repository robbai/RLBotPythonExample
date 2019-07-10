import os
import sys

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.collect_data import format_data, format_labels, data_size, label_size, from_labels
from teacher.teacher import Teacher
from util.dummy_renderer import DummyRenderer


class Learner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.last_time = 0

        # Teacher
        self.teacher = Teacher(self, self.team, self.index)
        self.teacher.initialize_agent()
        self.teacher.renderer = DummyRenderer(self.renderer)

        # Tensorflow
        import tensorflow as tf
        from tensorflow.keras import layers
        #self.tf = tf

        # Network
        regularisation_rate = 0.01
        self.model = tf.keras.Sequential([
        layers.Dense(data_size, activation = 'sigmoid', input_shape = (data_size,), kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),
        layers.Dense(data_size, activation = 'sigmoid', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),
        layers.Dense(label_size, activation = 'sigmoid', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))])
        self.model.compile(optimizer = tf.train.AdamOptimizer(0.01),
                           loss = 'mse')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        teacher_output = self.teacher.get_output(packet)
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct()).reshape((1, data_size))
        labels = format_labels(teacher_output).reshape((1, label_size))

        output = self.model.predict(data)[0]
        print(labels.tolist(), output.tolist())

        time = packet.game_info.seconds_elapsed
        #if time - self.last_time > 0.1:
        if True:
            self.train(data, labels)
            self.last_time= time
        
        self.controller_state = from_labels(output)
        return self.controller_state

    def train(self, data, labels):
        self.model.fit([data], [labels], epochs = 1)


