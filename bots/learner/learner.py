import os
import sys
from random import shuffle

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility.collect_data import format_data, format_labels, data_size, label_size, from_labels
from utility.dummy_renderer import DummyRenderer


game_speed = 1.6
dummy_render = False


class Learner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.last_save = 0
        self.state_set = False

        # Variables
        self.epochs = 25
        self.steps_used = None
        self.training_steps = None
        self.update_training_params()
        self.save_time = 600

        # Data and labels
        self.gathered_data = []
        self.gathered_labels = []

        # Teacher
        try:
            sys.path.append(r'C:/Users/wood3/Documents/RLBot/Bots/Dweller')
            from dweller import Dweller as Teacher
        except Exception as e:
            print(e)
            from teacher import Teacher
        self.teacher_name = Teacher.__name__
        self.teacher = Teacher(self, self.team, self.index)
        self.reset_teacher_functions(first_time = True)
        self.teacher.initialize_agent()
        
        # Tensorflow
        import tensorflow as tf
        from tensorflow.keras import layers
        #self.tf = tf

        # Network
        regularisation_rate = 0.00000001
        inputs = layers.Input(shape = (data_size,))
        x = layers.Dense(data_size, activation = 'linear', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))(inputs)
        x = layers.Dense(data_size, activation = 'linear', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))(x)
        output_one = layers.Dense(label_size[0], activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))(x)
        output_two = layers.Dense(label_size[1], activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))(x)
        self.model = tf.keras.Model(inputs = inputs, outputs = [output_one, output_two])
        self.model.compile(optimizer = tf.compat.v1.train.AdamOptimizer(0.0005), loss = ['mse', max_loss])

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        car = packet.game_cars[self.index]
        time = packet.game_info.seconds_elapsed

        # State-set the game speed
        if packet.game_info.is_round_active:
            if not self.state_set:
                game_state = GameState(console_commands=['Set WorldInfo TimeDilation ' + str(game_speed)])
                self.set_game_state(game_state)
                self.state_set = True
        else:
            self.state_set = False
        
        if not packet.game_info.is_round_active or car.is_demolished:
            return self.controller_state 
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct())
        labels = None

        # Get the labels
        self.reset_teacher_functions()
        teacher_output = self.teacher.get_output(packet)
        labels = format_labels(teacher_output, car)
        self.gathered_data.append(data)
        self.gathered_labels.append(labels)

        # Get our own predicted output
        output = self.model.predict(data.reshape((1, data_size)))

        # Train
        self.renderer.begin_rendering('Status')
        if len(self.gathered_data) >= self.training_steps:
            self.renderer.draw_string_2d(10, 10 + 100 * self.index, 2, 2, 'Training', self.renderer.team_color(car.team, True))
            self.renderer.end_rendering()

            # Randomise data and labels
            c = list(zip(self.gathered_data, self.gathered_labels))
            shuffle(c)
            data, labels = zip(*c)

            # Begin training
            steps = int(self.training_steps * self.steps_used)
            #self.train(data[:steps], labels[:steps])
            labels_to_use = [[x[0] for x in labels[:steps]], [x[1] for x in labels[:steps]]]
            self.train([data[:steps]], labels_to_use)

            self.gathered_data.clear()
            self.gathered_labels.clear()

            self.update_training_params(time)
        else:
            self.renderer.draw_string_2d(10, 10 * (self.index + 1), 2, 2, 'Playing ('\
                                         + str(int(len(self.gathered_data) / self.training_steps * 100))\
                                         + '%)', self.renderer.team_color(car.team))
            self.renderer.end_rendering()

        # Save model
        if time - self.last_save > self.save_time:
            self.last_save = time
            self.save()

        self.controller_state = from_labels(output)
        return self.controller_state

    def train(self, data, labels):
        for i in range(2):
            game_state = GameState(console_commands=['Pause'])
            self.set_game_state(game_state)
            if i == 0: self.model.fit(data, labels, epochs = self.epochs)

    def save(self):
        try:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
                                , 'models/' + str(int(self.last_save)) + "_" + self.teacher_name + '_model.h5')
            print('[' + self.name + '] Saving to: ' + str(path).replace('\\', '/'))
            self.model.save(str(path))
        except Exception as e:
            print(e)

    def reset_teacher_functions(self, first_time: bool = False):
        if dummy_render:
            self.teacher.renderer = DummyRenderer(self.renderer)
        else:
            self.teacher.renderer = self.renderer

        if first_time:
            self.teacher.get_field_info = self.get_field_info
            self.teacher.get_ball_prediction_struct = self.get_ball_prediction_struct
            self.teacher.send_quick_chat = self.send_quick_chat

    def update_training_params(self, time: float = 0):
        self.training_steps = min(1000, max(10, time // 2))
        #self.steps_used = max(0.3, 1 / max(1, time / 1000))
        #self.training_steps = 500
        self.steps_used = 1


def max_loss(predicted_y, desired_y):
    import tensorflow as tf
    return tf.reduce_max(tf.abs(predicted_y - desired_y), reduction_indices = [-1])


def cubic_activation(inputs):
    import tensorflow as tf
    linearness = 1 / 2.5
    return tf.add(inputs * linearness, tf.pow(inputs, 3)) / (1 + linearness)
