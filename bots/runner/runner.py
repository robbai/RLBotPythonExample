import os
import sys

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utility.collect_data import format_data, data_size, from_labels


model_extension = '.h5'
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/')


class Runner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        
        # Tensorflow
        import tensorflow as tf
        from tensorflow.keras import layers

        # Network
        model_name = find_model()
        self.model = tf.keras.models.load_model(model_path + model_name)
        print('[' + self.name + '] Loaded model: ' + str(model_path + model_name).replace('\\', '/'))

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        car = packet.game_cars[self.index]
        if not packet.game_info.is_round_active or car.is_demolished:
            return self.controller_state 
        
        data = format_data(self.index, packet, self.get_ball_prediction_struct())

        # Get our own predicted output
        output = self.model.predict(data.reshape((1, data_size)))
        
        self.controller_state = from_labels(output)
        return self.controller_state


# Find a model.
def find_model(manual_name: str = None):
    for i in range(2 if manual_name else 1):
        for file in os.listdir(model_path):
            if not file.endswith(model_extension): continue
            if i == 0 and manual_name and str(file) != manual_name: continue
            return file
    return None
