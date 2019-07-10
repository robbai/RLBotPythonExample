'''
BALL:
ball: (X, Y, Z) / pitch width
ball velocity normalised: X, Y, Z
ball velocity magnitude / pitch width

BALL PREDICTION:
ball prediction displacement: (X, Y, Z) / pitch width, each second for 3 seconds

CARS:
our car: (X, Y, Z) / pitch width
our car nose: X, Y, Z
local ball normalised: X, Y, Z
our ball distance / pitch width
our car boost / 100
car velocity magnitude / pitch width
car velocity forward component
is super sonic
has wheel contact
has double jumped
repeat all our car info for closest opponent to ball

MISC:
is kickoff
time since last touch, 10th log
is round active
'''

from math import log10 as log

import numpy as np
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo, GameInfo, BallInfo
from rlbot.utils.structures.ball_prediction_struct import BallPrediction
from rlbot.agents.base_agent import SimpleControllerState

from .vec import Vec3
from .constants import *
from .orientation import Orientation, relative_location
from .util import *


data_size = (3 + 3 + 1 + (3 * 3) + (3 + 3 + 3 + 7) * 2 + 3)
car_data_size = 16
label_size = 9


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
    ball_position = Vec3(ball.physics.location) # Rescale
    for i in range(3):
        frame = (i + 1) * 60
        predicted_location = prediction.slices[frame].physics.location
        displacement = (predicted_location - ball_position) / pitch_side_uu
        data[7 + i * 3] = displacement.x
        data[8 + i * 3] = displacement.y
        data[9 + i * 3] = displacement.z
    
    # Cars
    my_car = packet.game_cars[index]
    enemy_car = get_enemy_car(index, packet)
    for i, car in enumerate([my_car, enemy_car]):
        if not car: continue
        car_position = Vec3(car.physics.location) / pitch_side_uu
        data[16 + i * car_data_size] = car_position.x
        data[17 + i * car_data_size] = car_position.y
        data[18 + i * car_data_size] = car_position.z
        car_orientation = Orientation(car.physics.rotation)
        car_direction = car_orientation.forward.normalised()
        data[19 + i * car_data_size] = car_direction.x
        data[20 + i * car_data_size] = car_direction.y
        data[21 + i * car_data_size] = car_direction.z
        car_position = Vec3(car.physics.location) # Rescale
        local = relative_location(car_position, car_orientation, ball_position).normalised()
        data[22 + i * car_data_size] = local.x
        data[23 + i * car_data_size] = local.y
        data[24 + i * car_data_size] = local.z
        data[25 + i * car_data_size] = car_position.distance(ball_position) / pitch_side_uu
        data[26 + i * car_data_size] = car.boost / 100
        car_velocity_magnitude = car.physics.velocity.magnitude
        data[27 + i * car_data_size] = car_velocity_magnitude / pitch_side_uu
        data[28 + i * car_data_size] = car_direction.dot(car.physics.velocity) / car_velocity_magnitude
        data[29 + i * car_data_size] = (1 if car.is_super_sonic else -1)
        data[30 + i * car_data_size] = (1 if car.has_wheel_contact else -1)
        data[31 + i * car_data_size] = (1 if not car.double_jumped else -1)

    # Misc
    data[48] = (1 if packet.game_info.is_kickoff_pause else -1)
    data[49] = log(packet.game_info.seconds_elapsed - ball.latest_touch.time_seconds)
    data[50] = (1 if packet.game_info.is_round_active else -1)
    
    return data


def format_labels(controls: SimpleControllerState):
    labels = np.zeros(shape = label_size) # Blank labels
    labels[0] = controls.throttle
    labels[1] = controls.steer
    labels[2] = controls.pitch
    labels[3] = controls.yaw
    labels[4] = controls.roll
    labels[5] = (1 if controls.jump else -1)
    labels[6] = (1 if controls.boost else -1)
    labels[7] = (1 if controls.handbrake else -1)
    labels[8] = (1 if controls.use_item else -1)
    return labels


def from_labels(labels) -> SimpleControllerState:
    controls = SimpleControllerState()
    controls.throttle = clamp11(labels[0])
    controls.steer = clamp11(labels[1])
    controls.pitch = clamp11(labels[2])
    controls.yaw = clamp11(labels[3])
    controls.roll = clamp11(labels[4])
    controls.jump = (labels[5] > 0)
    controls.boost = (labels[6] > 0)
    controls.handbrake = (labels[7] > 0)
    controls.use_item = (labels[8] > 0)
    return controls

    
def get_enemy_car(index: int, packet: GameTickPacket) -> PlayerInfo:
    ball_position = Vec3(packet.game_ball.physics.location)
    team = (1 - packet.game_cars[index].team)

    closest_car = None
    min_distance = 0
    
    for index, car in enumerate(packet.game_cars[:packet.num_cars]):
        if not car or car.team != team: continue
        car_position = Vec3(car.physics.location)
        distance = car_position.distance(ball_position)

        if not closest_car or distance < min_distance:
            closest_car = car
            min_distance = distance

    return closest_car
