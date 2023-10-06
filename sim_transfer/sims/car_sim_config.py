from typing import Dict
import jax.numpy as jnp

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(jnp.array([-4.5, -4.5, -4., -2.5, -2.5, -1.]))

""" PARAMS FOR CAR 1 """

DEFAULT_PARAMS_BICYCLE_CAR1: Dict = {
    'use_blend': 0.0,
    'm': 1.65,
    'l_f': 0.13,
    'l_r': 0.17,
    'angle_offset': 0.0156,
    'b_f': 2.58,
    'b_r': 3.39,
    'blend_ratio_lb': 0.01,
    'blend_ratio_ub': 0.01,
    'c_d': 0.41464928,
    'c_f': 1.2,
    'c_m_1': 10.701814,
    'c_m_2': 1.4208151,
    'c_r': 1.27,
    'd_f': 0.02,
    'd_r': 0.017,
    'i_com': 0.01,
    'steering_limit': 0.3543
}

BOUNDS_PARAMS_BICYCLE_CAR1: Dict = {
    'use_blend': (0.0, 0.0),
    'm': (1.6, 1.7),
    'l_f': (0.11, 0.15),
    'l_r': (0.15, 0.19),
    'angle_offset': (0.001, 0.03),
    'b_f': (2.2, 2.8),
    'b_r': (2.0, 6.0),
    'blend_ratio_lb': (0.4, 0.4),
    'blend_ratio_ub': (0.5, 0.5),
    'c_d': (0.3, 0.5),
    'c_f': (1.2, 1.2),
    'c_m_1': (8., 13.),
    'c_m_2': (1.1, 1.7),
    'c_r': (1.27, 1.27),
    'd_f': (0.02, 0.02),
    'd_r': (0.017, 0.017),
    'i_com': (0.01, 0.1),
    'steering_limit': (0.20, 0.5),
}

DEFAULT_PARAMS_BLEND_CAR1: Dict = {
    'use_blend': 1.0,
    'm': 1.65,
    'l_f': 0.13,
    'l_r': 0.17,
    'angle_offset': -0.0213,
    'b_f': 1.8966477,
    'b_r': 6.2884626,
    'blend_ratio_lb': 0.06637411,
    'blend_ratio_ub': 0.00554,
    'c_d': 0.0,
    'c_f': 1.5381637,
    'c_m_1': 11.102413,
    'c_m_2': 1.3169205,
    'c_r': 1.186591,
    'd_f': 0.5968191,
    'd_r': 0.42716035,
    'i_com': 0.0685434,
    'steering_limit': 0.6337473,
}

BOUNDS_PARAMS_BLEND_CAR1 = {
    'use_blend': (1.0, 1.0),
    'm': (1.6, 1.7),
    'l_f': (0.125, 0.135),
    'l_r': (0.165, 0.175),
    'angle_offset': (-0.025, 0.025),
    'b_f': (1.3, 3.0),
    'b_r': (4.0, 10.0),
    'blend_ratio_lb': (0.01, 0.1),
    'blend_ratio_ub': (0.000, 0.2),
    'c_d': (0.0, 0.0),
    'c_f': (1.2, 1.8),
    'c_m_1': (10., 12.),
    'c_m_2': (1.1, 1.5),
    'c_r': (0.9, 1.5),
    'd_f': (0.35, 0.65),
    'd_r': (0.3, 0.6),
    'i_com': (0.05, 0.09),
    'steering_limit': (0.5, 0.9),
}

""" PARAMS FOR CAR 2 """

DEFAULT_PARAMS_BICYCLE_CAR2: Dict = {
    'use_blend': 0.0,
    'm': 1.65,
    'l_f': 0.13,
    'l_r': 0.17,
    'angle_offset': 0.031,
    'b_f': 2.58,
    'b_r': 4.75,
    'blend_ratio_lb': 0.01,
    'blend_ratio_ub': 0.01,
    'c_d': 0.0,
    'c_f': 1.2,
    'c_m_1': 8.46,
    'c_m_2': 1.6,
    'c_r': 1.27,
    'd_f': 0.02,
    'd_r': 0.017,
    'i_com': 0.01,
    'steering_limit': 0.347
}

BOUNDS_PARAMS_BICYCLE_CAR2: Dict = {
    'use_blend': (0.0, 0.0),
    'm': (1.6, 1.7),
    'l_f': (0.11, 0.15),
    'l_r': (0.15, 0.19),
    'angle_offset': (0.001, 0.05),
    'b_f': (2.2, 2.8),
    'b_r': (3.0, 7.0),
    'blend_ratio_lb': (0.4, 0.4),
    'blend_ratio_ub': (0.5, 0.5),
    'c_d': (0.01, 0.01),
    'c_f': (1.2, 1.2),
    'c_m_1': (6., 11.),
    'c_m_2': (1.1, 1.8),
    'c_r': (1.27, 1.27),
    'd_f': (0.02, 0.02),
    'd_r': (0.017, 0.017),
    'i_com': (0.01, 0.1),
    'steering_limit': (0.25, 0.45),
}

DEFAULT_PARAMS_BLEND_CAR2: Dict = {
    'use_blend': 1.0,
    'm': 1.65,
    'l_f': 0.13,
    'l_r': 0.17,
    'angle_offset': 0.02611047,
    'b_f': 2.5943623,
    'b_r': 5.2826314,
    'blend_ratio_lb': 0.0005,
    'blend_ratio_ub': 0.012,
    'c_d': 0.0,
    'c_f': 1.294,
    'c_m_1': 8.9,
    'c_m_2': 1.38,
    'c_r': 0.911,
    'd_f': 0.43,
    'd_r': 0.28,
    'i_com': 0.048,
    'steering_limit': 0.7,
}

BOUNDS_PARAMS_BLEND_CAR2 = {
    'use_blend': (1.0, 1.0),
    'm': (1.6, 1.7),
    'l_f': (0.125, 0.135),
    'l_r': (0.165, 0.175),
    'angle_offset': (0.0, 0.035),
    'b_f': (2.5, 3.5),
    'b_r': (4.0, 10.0),
    'blend_ratio_lb': (0.0001, 0.1),
    'blend_ratio_ub': (0.0001, 0.1),
    'c_d': (0.0, 0.0),
    'c_f': (1.1, 1.5),
    'c_m_1': (7., 10.),
    'c_m_2': (1.1, 1.5),
    'c_r': (0.4, 1.3),
    'd_f': (0.3, 0.6),
    'd_r': (0.15, 0.45),
    'i_com': (0.03, 0.07),
    'steering_limit': (0.6, 0.8),
}