'''
BALL:
ball: (X, Y, Z) / pitch width
ball velocity normalised: X, Y, Z
ball velocity magnitude / pitch width

CARS:
our car: (X, Y, Z) / pitch width
our car nose: X, Y, Z
local angular velocity: X, Y, Z
local ball normalised: X, Y, Z
local ball prediction normalised (each half second for 3 seconds): X, Y, Z
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
is round active
'''

from math import log10 as log

import numpy as np
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo, GameInfo, BallInfo, rotate_game_tick_packet_boost_omitted as flip_packet
from rlbot.utils.structures.ball_prediction_struct import BallPrediction
from rlbot.agents.base_agent import SimpleControllerState

from .vec import Vec3
from .constants import *
from .orientation import Orientation, relative_location
from .utility import *


opponent_data = False

data_size = (3 + 3 + 1 + (3 * 3) + (3 + 3 + 3 + 3 + 18 + 7) * (2 if opponent_data else 1) + 2)
car_data_size = 37
label_size = (5, 2, 1)


def format_data(index: int, packet: GameTickPacket, prediction: BallPrediction):
    data = np.zeros(shape = data_size) # Blank data

    flip = (packet.game_cars[index].team == 1) 
    if flip: flip_packet(packet)

    # Ball
    ball: BallData = packet.game_ball
    ball_position = Vec3(ball.physics.location) / pitch_side_uu
    ball_velocity = Vec3(ball.physics.velocity)
    ball_velocity_magnitude = ball_velocity.length() / pitch_side_uu
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
    
    
    # Cars
    my_car = packet.game_cars[index]
    enemy_car = (get_enemy_car(index, packet) if opponent_data else None)
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
        for j in range(6):
            frame = (j + 1) * 30
            predicted_location = Vec3(prediction.slices[frame].physics.location)
            if flip: predicted_location = Vec3(-predicted_location.x, -predicted_location.y, predicted_location.z)
            local = relative_location(car_position, car_orientation, predicted_location).normalised()
            data[25 + i * car_data_size + j * 3] = local.x
            data[26 + i * car_data_size + j * 3] = local.y
            data[27 + i * car_data_size + j * 3] = local.z
        data[43 + i * car_data_size] = car_position.dist(ball_position) / pitch_side_uu
        data[44 + i * car_data_size] = car.boost / 100
        car_velocity_magnitude = Vec3(car.physics.velocity).length()
        data[45 + i * car_data_size] = car_velocity_magnitude / pitch_side_uu
        data[46 + i * car_data_size] = car_direction.dot(Vec3(car.physics.velocity).normalised())
        data[47 + i * car_data_size] = (1 if car.is_super_sonic else -1)
        data[48 + i * car_data_size] = (1 if car.has_wheel_contact else -1)
        data[49 + i * car_data_size] = (1 if not car.double_jumped else -1)
        ang_vel = relative_location(Vec3(0, 0, 0), car_orientation, Vec3(car.physics.angular_velocity))
        data[50 + i * car_data_size] = ang_vel.x
        data[51 + i * car_data_size] = ang_vel.y
        data[52 + i * car_data_size] = ang_vel.z

    # Misc
    data[53 + (0 if not opponent_data else car_data_size)] = (1 if packet.game_info.is_kickoff_pause else -1)
    data[54 + (0 if not opponent_data else car_data_size)] = (1 if packet.game_info.is_round_active else -1)
    
    return data


def format_labels(controls: SimpleControllerState, car: PlayerInfo, mask: bool = False):
    labels = np.array([np.zeros(label_size[0]), np.zeros(label_size[1]), np.zeros(label_size[2])]) # Blank labels

    if mask:
        labels[0][0] = (1 if controls.boost and car.boost >= 1 else clamp11(controls.throttle))
        air: bool = (not car.has_wheel_contact or (controls.jump and not car.double_jumped))
        labels[0][1] = (clamp11(controls.steer) if not air else 0)
        labels[0][2] = (clamp11(controls.pitch) if air else 0)
        labels[0][3] = (clamp11(controls.yaw) if air else 0)
        labels[0][4] = (clamp11(controls.roll) if air else 0)
        labels[1][0] = (1 if controls.boost and car.boost >= 1 else -1)
        labels[1][1] = (1 if controls.handbrake and not air else -1)
        labels[2][0] = (1 if controls.jump and (air or not car.double_jumped) else -1)
    else:
        labels[0][0] = clamp11(controls.throttle)
        labels[0][1] = clamp11(controls.steer)
        labels[0][2] = clamp11(controls.pitch)
        labels[0][3] = clamp11(controls.yaw)
        labels[0][4] = clamp11(controls.roll)
        labels[1][0] = (1 if controls.boost else -1)
        labels[1][1] = (1 if controls.handbrake else -1)
        labels[2][0] = (1 if controls.jump else -1)
    
    return labels


def from_labels(labels) -> SimpleControllerState:
    controls = SimpleControllerState()
    labels = (labels[0][0], labels[1][0], labels[2][0])
    controls.throttle = clamp11(labels[0][0])
    controls.steer = clamp11(labels[0][1])
    controls.pitch = clamp11(labels[0][2])
    controls.yaw = clamp11(labels[0][3])
    controls.roll = clamp11(labels[0][4])
    controls.boost = (labels[1][0] > 0)
    controls.handbrake = (labels[1][1] > 0)
    controls.jump = (labels[2][0] > 0)
    controls.use_item = False
    return controls

    
def get_enemy_car(index: int, packet: GameTickPacket) -> PlayerInfo:
    ball_position = Vec3(packet.game_ball.physics.location)
    team = (1 - packet.game_cars[index].team)

    closest_car = None
    min_distance = 0
    
    for index, car in enumerate(packet.game_cars[:packet.num_cars]):
        if not car or car.team != team: continue
        car_position = Vec3(car.physics.location)
        distance = car_position.dist(ball_position)

        if not closest_car or distance < min_distance:
            closest_car = car
            min_distance = distance

    return closest_car
