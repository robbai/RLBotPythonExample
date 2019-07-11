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
        self.epochs = 1000
        self.step_size = 250
        self.play_on_own = False

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
        regularisation_rate = 0.1
        self.model = tf.keras.Sequential([\
        layers.Dense(data_size / 2, activation = 'sigmoid', input_shape = (data_size,), kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),\
        layers.Dense(data_size / 2, activation = 'sigmoid', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),\
        layers.Dense(label_size, activation = 'sigmoid', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))])
        self.model.compile(optimizer = tf.train.AdamOptimizer(0.01),\
                           loss = 'mse')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if not packet.game_info.is_round_active or packet.game_cars[self.index].is_demolished:
            return self.controller_state 
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct())
        labels = None

        if not self.play_on_own:
            self.reset_teacher_functions()
            teacher_output = self.teacher.get_output(packet)
            labels = format_labels(teacher_output)

            self.gathered_data.append(data)
            self.gathered_labels.append(labels)

        #print(len(self.gathered_data))

        output = self.model.predict(data.reshape((1, data_size)))[0]

        if (self.step_size is None or len(self.gathered_data) % self.step_size == 0)\
           and not self.play_on_own:
            c = list(zip(self.gathered_data, self.gathered_labels))
            shuffle(c)
            data, labels = zip(*c)
            
            self.train(data[:self.step_size], labels[:self.step_size])

            '''self.gathered_data.clear()
            self.gathered_labels.clear()'''
        
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
