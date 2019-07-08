'''
BALL:
ball: (X, Y, Z) / pitch width
ball velocity normalised: X, Y, Z
ball velocity magnitude / pitch width
ball prediction displacement: (X, Y, Z) / pitch width, each second for 3 seconds

CARS:
our car: X, Y, Z (scaled to pitch dimensions)
our car nose: X, Y, Z
local ball normalised: X, Y, Z
our ball distance / pitch width
our car boost / 100
car velocity magnitude / pitch width
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
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo, GameInfo, BallInfo
from rlbot.utils.structures.ball_prediction_struct import BallPrediction

from ..util.vec import Vec3

data_size = (3 + 3 + 1 + (3 * 3) + (3 + 3 + 3 + 7) * 2 + 3)


def format_data(index: int, packet: GameTickPacket, prediction: BallPrediction):
    data = np.zeros(shape = data_size) # Blank data

    return data
