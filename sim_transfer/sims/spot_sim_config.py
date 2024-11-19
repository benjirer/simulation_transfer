from typing import Dict
import jax.numpy as jnp

############################################ Learning Configs ############################################

################################ Sim Configs ################################
SPOT_DT = 1.0 / 15.0

SPOT_DOMAIN_LOWER = jnp.array(
    [
        # base pos
        -2.5,
        -2.5,
        -jnp.pi,
        # base vel
        -1.6,
        -1.6,
        -1.5,
        # ee pos
        -2.5,
        -2.5,
        0.1,
        # ee vel
        -2.5,
        -2.5,
        -2.5,
        # ee orientation
        -jnp.pi,
        -jnp.pi,
        -jnp.pi,
        # ee angular vel
        -2.5,
        -2.5,
        -2.5,
        # base action
        -1.6,
        -1.6,
        -1.5,
        # ee action
        -2.5,
        -2.5,
        -2.5,
        # ee angular action
        -2.5,
        -2.5,
        -2.5,
    ]
)

SPOT_DOMAIN_UPPER = jnp.array(
    [
        # base pos
        4.5,
        2.5,
        jnp.pi,
        # base vel
        1.6,
        1.6,
        1.5,
        # ee pos
        4.5,
        2.5,
        1.8,
        # ee vel
        2.5,
        2.5,
        2.5,
        # ee orientation
        jnp.pi,
        jnp.pi,
        jnp.pi,
        # ee angular vel
        2.5,
        2.5,
        2.5,
        # base action
        1.6,
        1.6,
        1.5,
        # ee action
        2.5,
        2.5,
        2.5,
        # ee angular action
        2.5,
        2.5,
        2.5,
    ]
)

################################ Angle Idx ################################
SPOT_ANGLE_IDX = [2, 12, 13, 14]

################################ Masks ################################
# state mask
SPOT_STATE_MASK = jnp.array(
    [
        1,  # base_x
        1,  # base_y
        1,  # base_theta
        1,  # base_vx
        1,  # base_vy
        1,  # base_vtheta
        1,  # ee_x
        1,  # ee_y
        1,  # ee_z
        1,  # ee_vx
        1,  # ee_vy
        1,  # ee_vz
        1,  # ee_rx
        1,  # ee_ry
        1,  # ee_rz
        1,  # ee_vrx
        1,  # ee_vry
        1,  # ee_vrz
    ]
)

SPOT_STATE_MASK_ENCODED = jnp.concatenate(
    [
        jnp.array([val, val]) if idx in SPOT_ANGLE_IDX else jnp.array([val])
        for idx, val in enumerate(SPOT_STATE_MASK)
    ]
)

# action mask
SPOT_ACTION_MASK = jnp.array(
    [
        1,  # base_vx
        1,  # base_vy
        1,  # base_vtheta
        1,  # ee_vx
        1,  # ee_vy
        1,  # ee_vz
        1,  # ee_vrx
        1,  # ee_vry
        1,  # ee_vrz
    ]
)

# goal mask
SPOT_GOAL_MASK = jnp.array(
    [
        1,  # ee_x
        1,  # ee_y
        1,  # ee_z
        1,  # ee_rx
        1,  # ee_ry
        1,  # ee_rz
    ]
)

################################ Lengths ################################
# SPOT_STATE_LENGTH = jnp.sum(SPOT_STATE_MASK)
# SPOT_STATE_LENGTH_ENCODED = jnp.sum(SPOT_STATE_MASK_ENCODED)
# SPOT_ACTION_LENGTH = jnp.sum(SPOT_ACTION_MASK)
# SPOT_GOAL_LENGTH = jnp.sum(SPOT_GOAL_MASK)
SPOT_STATE_LENGTH = 18
SPOT_STATE_LENGTH_ENCODED = 22
SPOT_ACTION_LENGTH = 9
SPOT_GOAL_LENGTH = 6

################################ Lables ################################
SPOT_STATE_LABELS_PRE = [
    "base_x",
    "base_y",
    "base_theta",
    "base_vx",
    "base_vy",
    "base_vtheta",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "ee_rx",
    "ee_ry",
    "ee_rz",
    "ee_vrx",
    "ee_vry",
    "ee_vrz",
]

SPOT_STATE_LABELS_ENCODED_PRE = [
    "base_x",
    "base_y",
    "sin_base_theta",
    "cos_base_theta",
    "base_vx",
    "base_vy",
    "base_vtheta",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "sin_ee_rx",
    "cos_ee_rx",
    "sin_ee_ry",
    "cos_ee_ry",
    "sin_ee_rz",
    "cos_ee_rz",
    "ee_vrx",
    "ee_vry",
    "ee_vrz",
]

SPOT_ACTION_LABELS_PRE = [
    "base_vx",
    "base_vy",
    "base_vtheta",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "ee_vrx",
    "ee_vry",
    "ee_vrz",
]

SPOT_GOAL_LABELS_PRE = [
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_rx",
    "ee_ry",
    "ee_rz",
]

# apply masks
SPOT_STATE_LABELS = [
    label for i, label in enumerate(SPOT_STATE_LABELS_PRE) if SPOT_STATE_MASK[i]
]
SPOT_STATE_LABELS_ENCODED = [
    label
    for i, label in enumerate(SPOT_STATE_LABELS_ENCODED_PRE)
    if SPOT_STATE_MASK_ENCODED[i]
]
SPOT_ACTION_LABELS = [
    label for i, label in enumerate(SPOT_ACTION_LABELS_PRE) if SPOT_ACTION_MASK[i]
]
SPOT_GOAL_LABELS = [
    label for i, label in enumerate(SPOT_GOAL_LABELS_PRE) if SPOT_GOAL_MASK[i]
]

############################################ Spot Params Collection ############################################

################################ Default Parameters ################################
# from alpha_betavel_set_5
SPOT_DEFAULT_PARAMS: Dict = {
    "alpha_base_1": 0.36060643,
    "alpha_base_2": 0.5552592,
    "alpha_base_3": 0.31399533,
    "alpha_ee_1": 0.03760632,
    "alpha_ee_2": -0.00998792,
    "alpha_ee_3": 0.4368394,
    "beta_base_1": 0.0,
    "beta_base_2": 0.0,
    "beta_base_3": 0.0,
    "beta_base_4": -0.00609878,
    "beta_base_5": -0.00455354,
    "beta_base_6": 0.00350315,
    "beta_ee_1": 0.0,
    "beta_ee_2": 0.0,
    "beta_ee_3": 0.0,
    "beta_ee_4": -0.0235504,
    "beta_ee_5": 0.0039239,
    "beta_ee_6": 0.00066999,
    "gamma_base_1": 1.0,
    "gamma_base_2": 1.0,
    "gamma_base_3": 1.0,
    "gamma_ee_1": 1.0,
    "gamma_ee_2": 1.0,
    "gamma_ee_3": 1.0,
}

SPOT_DEFAULT_OBSERVATION_NOISE_STD: jnp.array = 0.1 * jnp.exp(
    jnp.array(
        [
            -3.1841354,
            -3.6705942,
            -3.387062,
            -2.064331,
            -2.4769685,
            -1.7474595,
            -3.0793405,
            -3.1900687,
            -3.8907578,
            -1.6901332,
            -1.4695036,
            -2.0794985,
        ]
    )
)

SPOT_DEFAULT_PARAMS_WITH_EE_ORIENTATION: Dict = {
    "alpha_base_1": 0.36060643,
    "alpha_base_2": 0.5552592,
    "alpha_base_3": 0.31399533,
    "alpha_ee_1": 0.03760632,
    "alpha_ee_2": -0.00998792,
    "alpha_ee_3": 0.4368394,
    "beta_base_1": 0.0,
    "beta_base_2": 0.0,
    "beta_base_3": 0.0,
    "beta_base_4": -0.00609878,
    "beta_base_5": -0.00455354,
    "beta_base_6": 0.00350315,
    "beta_ee_1": 0.0,
    "beta_ee_2": 0.0,
    "beta_ee_3": 0.0,
    "beta_ee_4": -0.0235504,
    "beta_ee_5": 0.0039239,
    "beta_ee_6": 0.00066999,
    "gamma_base_1": 1.0,
    "gamma_base_2": 1.0,
    "gamma_base_3": 1.0,
    "gamma_ee_1": 1.0,
    "gamma_ee_2": 1.0,
    "gamma_ee_3": 1.0,
    # "alpha_base_1": 0.0,
    # "alpha_base_2": 0.0,
    # "alpha_base_3": 0.0,
    # "alpha_ee_1": 0.0,
    # "alpha_ee_2": 0.0,
    # "alpha_ee_3": 0.0,
    # "beta_base_1": 0.0,
    # "beta_base_2": 0.0,
    # "beta_base_3": 0.0,
    # "beta_base_4": 0.0,
    # "beta_base_5": 0.0,
    # "beta_base_6": 0.0,
    # "beta_ee_1": 0.0,
    # "beta_ee_2": 0.0,
    # "beta_ee_3": 0.0,
    # "beta_ee_4": 0.0,
    # "beta_ee_5": 0.0,
    # "beta_ee_6": 0.0,
    # "gamma_base_1": 1.0,
    # "gamma_base_2": 1.0,
    # "gamma_base_3": 1.0,
    # "gamma_ee_1": 1.0,
    # "gamma_ee_2": 1.0,
    # "gamma_ee_3": 1.0,
    # include EE orientation, TODO: need to get these from the new data
    "alpha_ee_ang_1": 0.3,
    "alpha_ee_ang_2": 0.3,
    "alpha_ee_ang_3": 0.3,
    "beta_ee_ang_1": 0.0,
    "beta_ee_ang_2": 0.0,
    "beta_ee_ang_3": 0.0,
    "beta_ee_ang_4": 0.0,
    "beta_ee_ang_5": 0.0,
    "beta_ee_ang_6": 0.0,
    "gamma_ee_ang_1": 1.0,
    "gamma_ee_ang_2": 1.0,
    "gamma_ee_ang_3": 1.0,
}

SPOT_DEFAULT_OBSERVATION_NOISE_STD_WITH_EE_ORIENTATION: jnp.array = 0.1 * jnp.exp(
    jnp.array(
        [
            -3.18,
            -3.67,
            -3.39,
            -2.06,
            -2.48,
            -1.75,
            -3.08,
            -3.19,
            -3.89,
            -1.69,
            -1.47,
            -2.08,
            -3.91,
            -3.91,
            -3.91,
            -1.71,
            -1.71,
            -1.71,
        ]
    )
)

################################ Parameter Bounds ################################
BOUNDS_SPOT_MODEL_PARAMS: Dict = {
    "alpha_base_1": (0.0, 0.8),
    "alpha_base_2": (0.0, 0.8),
    "alpha_base_3": (0.0, 0.8),
    "alpha_ee_1": (0.0, 0.2),
    "alpha_ee_2": (0.0, 0.2),
    "alpha_ee_3": (0.0, 0.6),
    "beta_base_1": (-0.003, 0.003),
    "beta_base_2": (-0.003, 0.003),
    "beta_base_3": (-0.003, 0.003),
    "beta_base_4": (-0.01, 0.01),
    "beta_base_5": (-0.01, 0.01),
    "beta_base_6": (-0.01, 0.01),
    "beta_ee_1": (-0.003, 0.003),
    "beta_ee_2": (-0.003, 0.003),
    "beta_ee_3": (-0.003, 0.003),
    "beta_ee_4": (-0.01, 0.01),
    "beta_ee_5": (-0.01, 0.01),
    "beta_ee_6": (-0.01, 0.01),
    "gamma_base_1": (0.9, 1.8),
    "gamma_base_2": (0.9, 1.8),
    "gamma_base_3": (0.9, 1.8),
    "gamma_ee_1": (0.9, 1.8),
    "gamma_ee_2": (0.9, 1.8),
    "gamma_ee_3": (0.9, 1.8),
}

BOUNDS_SPOT_MODEL_PARAMS_WITH_EE_ORIENTATION: Dict = {
    "alpha_base_1": (0.0, 0.8),
    "alpha_base_2": (0.0, 0.8),
    "alpha_base_3": (0.0, 0.8),
    "alpha_ee_1": (0.0, 0.2),
    "alpha_ee_2": (0.0, 0.2),
    "alpha_ee_3": (0.0, 0.6),
    "beta_base_1": (-0.003, 0.003),
    "beta_base_2": (-0.003, 0.003),
    "beta_base_3": (-0.003, 0.003),
    "beta_base_4": (-0.01, 0.01),
    "beta_base_5": (-0.01, 0.01),
    "beta_base_6": (-0.01, 0.01),
    "beta_ee_1": (-0.003, 0.003),
    "beta_ee_2": (-0.003, 0.003),
    "beta_ee_3": (-0.003, 0.003),
    "beta_ee_4": (-0.01, 0.01),
    "beta_ee_5": (-0.01, 0.01),
    "beta_ee_6": (-0.01, 0.01),
    "gamma_base_1": (0.9, 1.8),
    "gamma_base_2": (0.9, 1.8),
    "gamma_base_3": (0.9, 1.8),
    "gamma_ee_1": (0.9, 1.8),
    "gamma_ee_2": (0.9, 1.8),
    "gamma_ee_3": (0.9, 1.8),
    "alpha_ee_ang_1": (0.2, 0.8),
    "alpha_ee_ang_2": (0.2, 0.8),
    "alpha_ee_ang_3": (0.2, 0.8),
    "beta_ee_ang_1": (-0.003, 0.003),
    "beta_ee_ang_2": (-0.003, 0.003),
    "beta_ee_ang_3": (-0.003, 0.003),
    "beta_ee_ang_4": (-0.003, 0.003),
    "beta_ee_ang_5": (-0.003, 0.003),
    "beta_ee_ang_6": (-0.003, 0.003),
    "gamma_ee_ang_1": (0.95, 1.05),
    "gamma_ee_ang_2": (0.95, 1.05),
    "gamma_ee_ang_3": (0.95, 1.05),
}

################################ Normalization Stats ################################
# using all datasets
SPOT_MODEL_NORMALIZATION_STATS: Dict = {
    "x_mean": jnp.array(
        [
            0.887,
            0.148,
            0.06,
            0.01,
            0.003,
            -0.0,
            1.911,
            0.198,
            0.475,
            0.011,
            0.003,
            -0.002,
            0.02,
            -0.001,
            -0.003,
            0.028,
            -0.002,
            -0.007,
        ]
    ),
    "x_std": jnp.array(
        [
            0.994,
            0.332,
            0.204,
            0.436,
            0.191,
            0.249,
            1.021,
            0.39,
            0.2,
            0.459,
            0.321,
            0.184,
            0.438,
            0.157,
            0.232,
            0.194,
            0.207,
            0.192,
        ]
    ),
    "y_mean": jnp.array(
        [
            0.889,
            0.148,
            0.06,
            0.01,
            0.003,
            -0.0,
            1.913,
            0.198,
            0.475,
            0.011,
            0.003,
            -0.002,
        ]
    ),
    "y_std": jnp.array(
        [
            0.994,
            0.332,
            0.204,
            0.436,
            0.191,
            0.249,
            1.021,
            0.39,
            0.2,
            0.459,
            0.321,
            0.184,
        ]
    ),
}

SPOT_MODEL_NORMALIZATION_STATS_ENCODED_ANGLE: Dict = {
    "x_mean": jnp.array(
        [
            0.887,
            0.148,
            0.058,
            0.978,
            0.01,
            0.003,
            -0.0,
            1.911,
            0.198,
            0.475,
            0.011,
            0.003,
            -0.002,
            0.02,
            -0.001,
            -0.003,
            0.028,
            -0.002,
            -0.007,
        ]
    ),
    "x_std": jnp.array(
        [
            0.994,
            0.332,
            0.194,
            0.052,
            0.436,
            0.191,
            0.249,
            1.021,
            0.39,
            0.2,
            0.459,
            0.321,
            0.184,
            0.438,
            0.157,
            0.232,
            0.194,
            0.207,
            0.192,
        ]
    ),
    "y_mean": jnp.array(
        [
            0.889,
            0.148,
            0.058,
            0.978,
            0.01,
            0.003,
            -0.0,
            1.913,
            0.198,
            0.475,
            0.011,
            0.003,
            -0.002,
        ]
    ),
    "y_std": jnp.array(
        [
            0.994,
            0.332,
            0.194,
            0.052,
            0.436,
            0.191,
            0.249,
            1.021,
            0.39,
            0.2,
            0.459,
            0.321,
            0.184,
        ]
    ),
}

SPOT_MODEL_NORMALIZATION_STATS_WITH_EE_ORIENTATION: Dict = {
    "x_mean": jnp.array(
        [
            1.247,
            -0.365,
            0.104,
            0.02,
            -0.007,
            -0.005,
            1.925,
            -0.275,
            0.667,
            0.024,
            -0.009,
            -0.001,
            -0.047,
            0.109,
            0.055,
            -0.008,
            -0.007,
            0.038,
            0.059,
            -0.005,
            -0.016,
            0.21,
            -0.025,
            -0.178,
            -0.018,
            0.01,
            -0.028,
        ]
    ),
    "x_std": jnp.array(
        [
            0.542,
            0.319,
            0.342,
            0.504,
            0.307,
            0.461,
            0.631,
            0.411,
            0.275,
            0.513,
            0.471,
            0.391,
            1.941,
            0.569,
            1.287,
            1.284,
            1.273,
            1.292,
            0.682,
            0.409,
            0.57,
            1.091,
            1.289,
            1.437,
            1.461,
            1.449,
            1.455,
        ]
    ),
    "y_mean": jnp.array(
        [
            1.249,
            -0.366,
            0.104,
            0.021,
            -0.007,
            -0.006,
            1.926,
            -0.276,
            0.668,
            0.025,
            -0.009,
            -0.001,
            -0.047,
            0.109,
            0.056,
            -0.009,
            -0.007,
            0.039,
        ]
    ),
    "y_std": jnp.array(
        [
            0.541,
            0.319,
            0.342,
            0.505,
            0.308,
            0.461,
            0.63,
            0.412,
            0.275,
            0.514,
            0.472,
            0.392,
            1.942,
            0.569,
            1.288,
            1.285,
            1.274,
            1.292,
        ]
    ),
}

SPOT_MODEL_NORMALIZATION_STATS_WITH_EE_ORIENTATION_ENCODED_ANGLE: Dict = {
    "x_mean": jnp.array(
        [
            2.44,
            0.313,
            0.13,
            0.873,
            0.013,
            0.0,
            0.004,
            3.073,
            0.464,
            0.772,
            0.012,
            0.001,
            -0.003,
            0.002,
            -0.043,
            0.178,
            0.792,
            0.227,
            0.099,
            0.008,
            -0.017,
            -0.001,
            0.034,
            0.015,
            -0.001,
            0.009,
            0.006,
            -0.008,
            -0.011,
            0.003,
            -0.022,
        ]
    ),
    "x_std": jnp.array(
        [
            1.425,
            0.399,
            0.355,
            0.307,
            0.468,
            0.314,
            0.428,
            1.413,
            0.503,
            0.304,
            0.492,
            0.403,
            0.204,
            0.714,
            0.699,
            0.542,
            0.22,
            0.665,
            0.705,
            1.032,
            1.013,
            0.991,
            0.637,
            0.388,
            0.43,
            0.182,
            0.196,
            0.22,
            1.349,
            1.358,
            1.352,
        ]
    ),
    "y_mean": jnp.array(
        [
            2.442,
            0.313,
            0.13,
            0.873,
            0.013,
            0.0,
            0.004,
            3.075,
            0.464,
            0.772,
            0.012,
            0.001,
            -0.003,
            0.002,
            -0.044,
            0.178,
            0.791,
            0.226,
            0.099,
            0.008,
            -0.017,
            -0.001,
        ]
    ),
    "y_std": jnp.array(
        [
            1.424,
            0.399,
            0.355,
            0.307,
            0.468,
            0.314,
            0.428,
            1.412,
            0.503,
            0.304,
            0.493,
            0.403,
            0.204,
            0.714,
            0.699,
            0.542,
            0.22,
            0.665,
            0.705,
            1.032,
            1.013,
            0.991,
        ]
    ),
}
