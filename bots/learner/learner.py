from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from ..utils.collect_data import format_data, format_labels, data_size, label_size
from ..teacher.teacher import Teacher
from ..utils.dummy_renderer import DummyRenderer


class Learner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()

        # Teacher
        self.teacher = Teacher(self)
        self.teacher.renderer = DummyRenderer(self.renderer)

        # Tensorflow
        import tensorflow as tf
        from tensorflow import keras
        #self.tf = tf
        #self.keras = keras

        # Network
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape = data_size),
            keras.layers.Dense(data_size, activation = tf.nn.relu),
            keras.layers.Dense(label_size, activation = tf.nn.relu)
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        teacher_output = self.teacher.get_output(packet)
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct())
        labels = format_labels(teacher_output)

        output = self.model.predict([data])
        self.controller_state = from_labels(output)

        self.train(data, labels)
        
        return self.controller_state

    def train(self, data, labels):
        self.model.fit([data], [labels], epochs = 1)


