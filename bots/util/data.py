'''
BALL:
ball: (X, Y, Z) / pitch width
ball velocity normalised: X, Y, Z
ball velocity magnitude / pitch width

BALL PREDICTION:
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
from ..util.constants import *

data_size = (3 + 3 + 1 + (3 * 3) + (3 + 3 + 3 + 7) * 2 + 3)


def format_data(index: int, packet: GameTickPacket, prediction: BallPrediction):
    data = np.zeros(shape = data_size) # Blank data

    # Ball
    ball: BallData = packet.game_ball
    ball_position = Vec3(ball.physics.location) / pitch_side_uu
    ball_velocity = Vec3(ball.physics.velocity)
    ball_velocity_magnitude = ball_velocity.magnitude / pitch_side_uu
    ball_velocity = ball_velocity.normalised()
    data[0] = ball_position.x
    data[1] = ball_position.y
    data[2] = ball_position.z
    data[3] = ball_velocity.x
    data[4] = ball_velocity.y
    data[5] = ball_velocity.z
    data[6] = ball_velocity_magnitude

    # Ball prediction
    ball_position = Vec3(ball.physics.location)
    for i in range(3):
        frame = (i + 1) * 60
        predicted_location = prediction.slices[frame].physics.location
        displacement = (predicted_location - ball_position) / pitch_side_uu
        data[7 + i * 3] #TODO
    

    return data
