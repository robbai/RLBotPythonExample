from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from ..utils.collect_data import format_data
from ..teacher.teacher import Teacher
from ..utils.dummy_renderer import DummyRenderer


class Learner(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.teacher = Teacher(self)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        data = format_data(self.index, packet, self.get_ball_prediction_struct())
        
        return self.controller_state


