'''
BALL:
ball: X, Y, Z (scaled to pitch dimensions)
ball velocity normalised: X, Y, Z
ball velocity magnitude / pitch length
ball prediction displacement: X, Y, Z (scaled to pitch dimensions, each 0.5 seconds for 3 seconds)

CARS:
our car: X, Y, Z (scaled to pitch dimensions)
local ball normalised: X, Y, Z
our ball distance / pitch length
our car boost / 100
car velocity magnitude / pitch length
car velocity forward component
has wheel contact
is super sonic
has double jumped
repeat all our car info for closest opponent to ball

MISC:
is kickoff
time since last touch, 10th log
is round active
'''

import numpy as np
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import BallPrediction


data_size = (3 + 3 + 1 + 6 + (3 + 3 + 7) * 2 + 3)


def format_data(packet: GameTickPacket, prediction: BallPrediction):
    data = np.zeros(shape = data_size) # Blank data


    return data
