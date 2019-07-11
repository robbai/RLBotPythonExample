import os
import sys
from random import shuffle

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility.collect_data import format_data, format_labels, data_size, label_size, from_labels
from utility.dummy_renderer import DummyRenderer


dummy_render = False


class Learner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.last_time = 0

        # Variables
        self.epochs = 5
        self.step_size = 30
        self.play_on_own = False
        self.max_data_size = 500

        # Data and labels
        self.gathered_data = []
        self.gathered_labels = []

        # Teacher
        try:
            sys.path.append(r'C:/Users/wood3/Documents/RLBot/Bots/Atba2')
            from atba2 import Atba2 as Teacher
        except Exception as e:
            print(e)
            from teacher import Teacher
        self.teacher = Teacher(self, self.team, self.index)
        self.reset_teacher_functions(first_time = True)
        self.teacher.initialize_agent()
        
        # Tensorflow
        import tensorflow as tf
        from tensorflow.keras import layers
        #self.tf = tf

        # Network
        regularisation_rate = 0.001
        self.model = tf.keras.Sequential([\
        layers.Dense(data_size, activation = 'tanh', input_shape = (data_size,), kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),\
        layers.Dense(data_size, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),\
        layers.Dense(label_size, activation = 'linear', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))])
        self.model.compile(optimizer = tf.train.AdamOptimizer(0.01),\
                           loss = 'mse')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        car = packet.game_cars[self.index]
        if not packet.game_info.is_round_active or car.is_demolished:
            return self.controller_state 
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct())
        labels = None

        # Get the labels
        if not self.play_on_own:
            self.reset_teacher_functions()
            teacher_output = self.teacher.get_output(packet)
            labels = format_labels(teacher_output, car.has_wheel_contact)

            self.gathered_data.append(data)
            self.gathered_labels.append(labels)

        # Get our own predicted output
        output = self.model.predict(data.reshape((1, data_size)))[0]

        # Train
        if (self.step_size is None or len(self.gathered_data) % self.step_size == 0)\
           and not self.play_on_own:
            c = list(zip(self.gathered_data, self.gathered_labels))
            shuffle(c)
            data, labels = zip(*c)
            
            self.train(data[:self.step_size], labels[:self.step_size])

            if self.max_data_size and len(self.gathered_data) > self.max_data_size:
                for i in range(len(self.gathered_data) - self.max_data_size):
                    del self.gathered_data[0]
                    del self.gathered_labels[0]
        
        self.controller_state = from_labels(output)
        return self.controller_state

    def train(self, data, labels):
        self.model.fit([data], [labels], epochs = self.epochs)

    def reset_teacher_functions(self, first_time: bool = False):
        if dummy_render:
            self.teacher.renderer = DummyRenderer(self.renderer)
        else:
            self.teacher.renderer = self.renderer

        if first_time:
            self.teacher.get_field_info = self.get_field_info
            self.teacher.get_ball_prediction_struct = self.get_ball_prediction_struct
            self.teacher.send_quick_chat = self.send_quick_chat
