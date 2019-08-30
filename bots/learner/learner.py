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


game_speed = 3
dummy_render = False


class Learner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.last_save = 0
        self.state_set = False

        # Variables
        self.epochs = 30
        self.steps_used = 0.25
        self.training_steps = 10
        self.play_on_own = False
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
        regularisation_rate = 0.0000001
        self.model = tf.keras.Sequential([\
        layers.Dense(data_size, activation = 'linear', input_shape = (data_size,), kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),\
        layers.Dense(data_size, activation = 'linear', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate)),\
        layers.Dense(label_size, activation = 'tanh', kernel_regularizer = tf.keras.regularizers.l2(l = regularisation_rate))])
        self.model.compile(optimizer = tf.train.AdamOptimizer(0.001),\
                           loss = 'mse')

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        car = packet.game_cars[self.index]
        time = packet.game_info.seconds_elapsed

        # State-set the game speed
        if not self.state_set:
            game_state = GameState(console_commands=['Set WorldInfo TimeDilation ' + str(game_speed)])
            self.set_game_state(game_state)
            self.state_set = True
        
        if not packet.game_info.is_round_active or car.is_demolished:
            return self.controller_state 
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct())
        labels = None

        # Get the labels
        if not self.play_on_own:
            self.reset_teacher_functions()
            teacher_output = self.teacher.get_output(packet)
            labels = format_labels(teacher_output, car)

            self.gathered_data.append(data)
            self.gathered_labels.append(labels)

        # Get our own predicted output
        output = self.model.predict(data.reshape((1, data_size)))[0]

        # Train
        self.renderer.begin_rendering('Status')
        if len(self.gathered_data) >= self.training_steps\
           and not self.play_on_own:
            self.renderer.draw_string_2d(10, 10 + 100 * self.index, 2, 2, 'Training', self.renderer.team_color(car.team, True))
            self.renderer.end_rendering()

            # Randomise data and labels
            c = list(zip(self.gathered_data, self.gathered_labels))
            shuffle(c)
            data, labels = zip(*c)

            # Begin training
            steps = int(self.training_steps * self.steps_used)
            self.train(data[:steps], labels[:steps])

            self.gathered_data.clear()
            self.gathered_labels.clear()

            self.update_training_params(time)
        else:
            self.renderer.draw_string_2d(10, 10 * (self.index + 1), 2, 2, 'Playing ('\
                                         + str(int(len(self.gathered_data) / self.training_steps * 100))\
                                         + '%)', self.renderer.team_color(car.team))
            self.renderer.end_rendering()

        # Save model
        if not self.play_on_own and time - self.last_save > self.save_time:
            self.last_save = time
            self.save()
        
        self.controller_state = from_labels(output)
        return self.controller_state

    def train(self, data, labels):
        self.model.fit([data], [labels], epochs = self.epochs)

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

    def update_training_params(self, time: float):
        self.training_steps = min(300, max(10, int(time)))
        self.steps_used = max(0.2, 1 / max(1, time / 150))
