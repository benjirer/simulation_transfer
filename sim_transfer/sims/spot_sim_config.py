from typing import Dict
import jax.numpy as jnp

############################################ Spot Params Collection ############################################

################################ Default Parameters ################################
# from alpha_set_3

SPOT_DEFAULT_PARAMS: Dict = {
    "alpha_base_1": 0.57470965,
    "alpha_base_2": 0.6886134,
    "alpha_base_3": -0.12433565,
    "alpha_ee_1": -0.04858014,
    "alpha_ee_2": 0.02063864,
    "alpha_ee_3": -0.4303356,
    "beta_base_1": 0.0,
    "beta_base_2": 0.0,
    "beta_base_3": 0.0,
    "beta_base_4": 0.0,
    "beta_base_5": 0.0,
    "beta_base_6": 0.0,
    "beta_ee_1": 0.0,
    "beta_ee_2": 0.0,
    "beta_ee_3": 0.0,
    "beta_ee_4": 0.0,
    "beta_ee_5": 0.0,
    "beta_ee_6": 0.0,
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
            -3.7909253,
            -3.6883085,
            -3.5094566,
            -2.3280942,
            -2.4077191,
            -1.579626,
            -3.5002875,
            -3.2951872,
            -3.7636478,
            -1.7756749,
            -1.5336628,
            -2.0430653,
        ]
    )
)

################################ Set 1 ################################

spot_model_alpha_set_1: Dict = {
    "alpha_base_1": 0.70359606,
    "alpha_base_2": 0.7651408,
    "alpha_base_3": 0.46212456,
    "alpha_ee_1": 0.13103251,
    "alpha_ee_2": 0.16283108,
    "alpha_ee_3": 0.11467764,
}

spot_model_gamma_set_1: Dict = {
    "gamma_base_1": 0.95941824,
    "gamma_base_2": 0.92387575,
    "gamma_base_3": 1.0936635,
    "gamma_ee_1": 1.0584811,
    "gamma_ee_2": 1.0163194,
    "gamma_ee_3": 1.3609403,
}

spot_model_alpha_betapos_set_1: Dict = {
    "alpha_base_1": 0.70348245,
    "alpha_base_2": 0.7633575,
    "alpha_base_3": 0.46046728,
    "alpha_ee_1": 0.13248844,
    "alpha_ee_2": 0.16524743,
    "alpha_ee_3": 0.11535439,
    "beta_base_1": -0.0021394,
    "beta_base_2": -0.00246076,
    "beta_base_3": 0.00274849,
    "beta_ee_1": -0.00411335,
    "beta_ee_2": -0.00041461,
    "beta_ee_3": 0.00139919,
}

spot_model_alpha_betavel_set_1: Dict = {
    "alpha_base_1": 0.70303154,
    "alpha_base_2": 0.757232,
    "alpha_base_3": 0.4605006,
    "alpha_ee_1": 0.13236454,
    "alpha_ee_2": 0.16600914,
    "alpha_ee_3": 0.11194759,
    "beta_base_4": 0.00184914,
    "beta_base_5": -0.00925563,
    "beta_base_6": 0.01103829,
    "beta_ee_4": -0.00476445,
    "beta_ee_5": -0.01713833,
    "beta_ee_6": -0.00798901,
}

spot_model_alpha_gamma_set_1: Dict = {
    "alpha_base_1": 0.7175345,
    "alpha_base_2": 0.7626418,
    "alpha_base_3": 0.48464295,
    "alpha_ee_1": 0.11278273,
    "alpha_ee_2": 0.11029007,
    "alpha_ee_3": 0.15383887,
    "gamma_base_1": 1.3176929,
    "gamma_base_2": 1.6753602,
    "gamma_base_3": 1.3003393,
    "gamma_ee_1": 1.289688,
    "gamma_ee_2": 1.2746578,
    "gamma_ee_3": 1.3894398,
}

spot_model_alpha_betapos_gamma_set_1: Dict = {
    "alpha_base_1": 0.7181579,
    "alpha_base_2": 0.7582589,
    "alpha_base_3": 0.48299763,
    "alpha_ee_1": 0.1136694,
    "alpha_ee_2": 0.11217932,
    "alpha_ee_3": 0.15967979,
    "beta_base_1": -0.00188543,
    "beta_base_2": -0.00345939,
    "beta_base_3": 0.00256174,
    "beta_ee_1": -0.00402828,
    "beta_ee_2": -0.00081502,
    "beta_ee_3": 0.00180176,
    "gamma_base_1": 1.326491,
    "gamma_base_2": 1.703256,
    "gamma_base_3": 1.2956551,
    "gamma_ee_1": 1.2964774,
    "gamma_ee_2": 1.2768157,
    "gamma_ee_3": 1.4050621,
}

spot_model_all_set_1: Dict = {
    "alpha_base_1": 0.7176426,
    "alpha_base_2": 0.75287974,
    "alpha_base_3": 0.48191804,
    "alpha_ee_1": 0.11441838,
    "alpha_ee_2": 0.11556692,
    "alpha_ee_3": 0.15769504,
    "beta_base_1": -0.00206197,
    "beta_base_2": -0.00291543,
    "beta_base_3": 0.00219428,
    "beta_base_4": 0.00305334,
    "beta_base_5": -0.00897289,
    "beta_base_6": 0.01028873,
    "beta_ee_1": -0.00427054,
    "beta_ee_2": -0.00060144,
    "beta_ee_3": 0.00189116,
    "beta_ee_4": -0.00311346,
    "beta_ee_5": -0.01762013,
    "beta_ee_6": -0.00775472,
    "gamma_base_1": 1.3262271,
    "gamma_base_2": 1.707615,
    "gamma_base_3": 1.2951155,
    "gamma_ee_1": 1.2955575,
    "gamma_ee_2": 1.2721424,
    "gamma_ee_3": 1.4047127,
}

################################ Set 2 ################################
# Addtional datasets, no action delay, no weights

spot_model_alpha_set_2: Dict = {
    "alpha_base_1": 0.6462474,
    "alpha_base_2": 0.8014192,
    "alpha_base_3": 0.57597494,
    "alpha_ee_1": 0.17684086,
    "alpha_ee_2": 0.14295992,
    "alpha_ee_3": 0.34643495,
}

spot_model_gamma_set_2: Dict = {
    "gamma_base_1": 1.0603073,
    "gamma_base_2": 0.36243778,
    "gamma_base_3": 0.82461816,
    "gamma_ee_1": 0.9885899,
    "gamma_ee_2": 0.6661898,
    "gamma_ee_3": 1.288408,
}

spot_model_alpha_betapos_set_2: Dict = {
    "alpha_base_1": 0.6454329,
    "alpha_base_2": 0.8009805,
    "alpha_base_3": 0.57372737,
    "alpha_ee_1": 0.18046111,
    "alpha_ee_2": 0.1452842,
    "alpha_ee_3": 0.346344,
    "beta_base_1": -0.00068514,
    "beta_base_2": -0.00117587,
    "beta_base_3": -0.00060223,
    "beta_ee_1": -0.00255925,
    "beta_ee_2": -0.00312724,
    "beta_ee_3": -0.00020149,
}

spot_model_alpha_betavel_set_2: Dict = {
    "alpha_base_1": 0.64413637,
    "alpha_base_2": 0.8005853,
    "alpha_base_3": 0.5721091,
    "alpha_ee_1": 0.18136305,
    "alpha_ee_2": 0.14406444,
    "alpha_ee_3": 0.3429685,
    "beta_base_4": -0.0048933,
    "beta_base_5": -0.00453804,
    "beta_base_6": -0.00223098,
    "beta_ee_4": -0.01858801,
    "beta_ee_5": -0.01230789,
    "beta_ee_6": -0.01333413,
}

spot_model_alpha_gamma_set_2: Dict = {
    "alpha_base_1": 0.6548139,
    "alpha_base_2": 0.8060915,
    "alpha_base_3": 0.5856956,
    "alpha_ee_1": 0.15350233,
    "alpha_ee_2": 0.1007416,
    "alpha_ee_3": 0.38865834,
    "gamma_base_1": 1.4514216,
    "gamma_base_2": 1.4367577,
    "gamma_base_3": 1.3156431,
    "gamma_ee_1": 1.2132027,
    "gamma_ee_2": 1.2396779,
    "gamma_ee_3": 1.4382231,
}

spot_model_alpha_betapos_gamma_set_2: Dict = {
    "alpha_base_1": 0.65381217,
    "alpha_base_2": 0.8056423,
    "alpha_base_3": 0.5834784,
    "alpha_ee_1": 0.15563801,
    "alpha_ee_2": 0.09956831,
    "alpha_ee_3": 0.388439,
    "beta_base_1": -0.00214883,
    "beta_base_2": -0.0020683,
    "beta_base_3": -0.0004636,
    "beta_ee_1": -0.003598,
    "beta_ee_2": -0.00392406,
    "beta_ee_3": -0.00023322,
    "gamma_base_1": 1.4572757,
    "gamma_base_2": 1.450601,
    "gamma_base_3": 1.3152777,
    "gamma_ee_1": 1.228738,
    "gamma_ee_2": 1.2546647,
    "gamma_ee_3": 1.437963,
}

spot_model_all_set_2: Dict = {
    "alpha_base_1": 0.6522914,
    "alpha_base_2": 0.8049851,
    "alpha_base_3": 0.58066773,
    "alpha_ee_1": 0.15855512,
    "alpha_ee_2": 0.10107709,
    "alpha_ee_3": 0.3857154,
    "beta_base_1": -0.00189418,
    "beta_base_2": -0.0018416,
    "beta_base_3": -0.00048653,
    "beta_base_4": -0.00453756,
    "beta_base_5": -0.00317156,
    "beta_base_6": -0.00096649,
    "beta_ee_1": -0.00325116,
    "beta_ee_2": -0.00355802,
    "beta_ee_3": 0.00015557,
    "beta_ee_4": -0.01666748,
    "beta_ee_5": -0.01270854,
    "beta_ee_6": -0.01269116,
    "gamma_base_1": 1.4572599,
    "gamma_base_2": 1.4503821,
    "gamma_base_3": 1.3145043,
    "gamma_ee_1": 1.2268625,
    "gamma_ee_2": 1.2484291,
    "gamma_ee_3": 1.4373773,
}


################################ Set 3 ################################
# Addtional datasets, with action delay, no weights

spot_model_all_set_3: Dict = {
    "alpha_base_1": 0.54686046,
    "alpha_base_2": 0.72122586,
    "alpha_base_3": 0.18406443,
    "alpha_ee_1": -0.00267955,
    "alpha_ee_2": -0.16879812,
    "alpha_ee_3": -0.08503822,
    "beta_base_1": -0.0024276,
    "beta_base_2": -0.00230496,
    "beta_base_3": -0.00083828,
    "beta_base_4": -0.00644391,
    "beta_base_5": -0.00658266,
    "beta_base_6": -0.00357458,
    "beta_ee_1": -0.00312369,
    "beta_ee_2": -0.00442108,
    "beta_ee_3": -0.00143356,
    "beta_ee_4": -0.01072194,
    "beta_ee_5": -0.01577909,
    "beta_ee_6": -0.01767765,
    "gamma_base_1": 1.3140954,
    "gamma_base_2": 1.3405886,
    "gamma_base_3": 1.2349811,
    "gamma_ee_1": 1.2321234,
    "gamma_ee_2": 1.3576127,
    "gamma_ee_3": 1.3250195,
}

spot_model_alpha_beta_pos_set_3: Dict = {
    "alpha_base_1": 0.5377294,
    "alpha_base_2": 0.70831037,
    "alpha_base_3": 0.16965973,
    "alpha_ee_1": 0.04929081,
    "alpha_ee_2": -0.0727505,
    "alpha_ee_3": -0.14128292,
    "beta_base_1": -0.0013653,
    "beta_base_2": -0.00176889,
    "beta_base_3": -0.00083792,
    "beta_ee_1": -0.0023359,
    "beta_ee_2": -0.00351499,
    "beta_ee_3": -0.00097217,
}

spot_model_alpha_beta_pos_beta_vel_set_3: Dict = {
    "alpha_base_1": 0.5333053,
    "alpha_base_2": 0.70640767,
    "alpha_base_3": 0.16692705,
    "alpha_ee_1": 0.05129206,
    "alpha_ee_2": -0.07274906,
    "alpha_ee_3": -0.14541936,
    "beta_base_1": -0.00122251,
    "beta_base_2": -0.00149219,
    "beta_base_3": -0.00081767,
    "beta_base_4": -0.00648582,
    "beta_base_5": -0.00703381,
    "beta_base_6": -0.00355876,
    "beta_ee_1": -0.00220113,
    "beta_ee_2": -0.00324125,
    "beta_ee_3": -0.0010863,
    "beta_ee_4": -0.01287764,
    "beta_ee_5": -0.01588303,
    "beta_ee_6": -0.01838979,
}

spot_model_alpha_beta_pos_gamma_set_3: Dict = {
    "alpha_base_1": 0.5503834,
    "alpha_base_2": 0.7228438,
    "alpha_base_3": 0.18620901,
    "alpha_ee_1": -0.00462879,
    "alpha_ee_2": -0.16981158,
    "alpha_ee_3": -0.08003535,
    "beta_base_1": -0.00262753,
    "beta_base_2": -0.00266248,
    "beta_base_3": -0.00086986,
    "beta_ee_1": -0.00327256,
    "beta_ee_2": -0.00470324,
    "beta_ee_3": -0.00134624,
    "gamma_base_1": 1.3141282,
    "gamma_base_2": 1.3408881,
    "gamma_base_3": 1.23537,
    "gamma_ee_1": 1.2337052,
    "gamma_ee_2": 1.3607633,
    "gamma_ee_3": 1.3255672,
}

spot_model_alpha_beta_vel_set_3: Dict = {
    "alpha_base_1": 0.5356108,
    "alpha_base_2": 0.7075563,
    "alpha_base_3": 0.1722591,
    "alpha_ee_1": 0.04929598,
    "alpha_ee_2": -0.0742145,
    "alpha_ee_3": -0.14143881,
    "beta_base_4": -0.00669486,
    "beta_base_5": -0.00872372,
    "beta_base_6": -0.00495519,
    "beta_ee_4": -0.01271631,
    "beta_ee_5": -0.01413471,
    "beta_ee_6": -0.01813824,
}

spot_model_alpha_beta_vel_gamma_set_3: Dict = {
    "alpha_base_1": 0.54988676,
    "alpha_base_2": 0.72230655,
    "alpha_base_3": 0.19035323,
    "alpha_ee_1": -0.00180756,
    "alpha_ee_2": -0.16343307,
    "alpha_ee_3": -0.07788584,
    "beta_base_4": -0.00760491,
    "beta_base_5": -0.00985584,
    "beta_base_6": -0.00627238,
    "beta_ee_4": -0.00987195,
    "beta_ee_5": -0.01045589,
    "beta_ee_6": -0.01736952,
    "gamma_base_1": 1.3113967,
    "gamma_base_2": 1.3319724,
    "gamma_base_3": 1.2359097,
    "gamma_ee_1": 1.2208637,
    "gamma_ee_2": 1.337389,
    "gamma_ee_3": 1.3245624,
}

spot_model_alpha_gamma_set_3: Dict = {
    "alpha_base_1": 0.5536129,
    "alpha_base_2": 0.7240096,
    "alpha_base_3": 0.19139135,
    "alpha_ee_1": -0.00542789,
    "alpha_ee_2": -0.1651529,
    "alpha_ee_3": -0.07781184,
    "gamma_base_1": 1.3096311,
    "gamma_base_2": 1.3288447,
    "gamma_base_3": 1.2360619,
    "gamma_ee_1": 1.22129,
    "gamma_ee_2": 1.3376801,
    "gamma_ee_3": 1.3248262,
}

spot_model_alpha_set_3: Dict = {
    "alpha_base_1": 0.54027677,
    "alpha_base_2": 0.70964324,
    "alpha_base_3": 0.17452711,
    "alpha_ee_1": 0.04534381,
    "alpha_ee_2": -0.07568604,
    "alpha_ee_3": -0.14013563,
}

spot_model_gamma_set_3: Dict = {
    "gamma_base_1": 1.2253829,
    "gamma_base_2": 0.87031513,
    "gamma_base_3": 1.2076699,
    "gamma_ee_1": 1.139749,
    "gamma_ee_2": 0.9694493,
    "gamma_ee_3": 1.3381345,
}

################################ Set 4 ################################
# Addtional datasets, with action delay, with weights

spot_model_alpha_set_4: Dict = {
    "alpha_base_1": 0.5589467,
    "alpha_base_2": 0.69328326,
    "alpha_base_3": -0.6078941,
    "alpha_ee_1": 0.15770227,
    "alpha_ee_2": 0.2367851,
    "alpha_ee_3": -0.79256815,
}

################################ Parameter Bounds ################################

bounds_spot_model_params: Dict = {
    "alpha_base_1": (-0.5, 1.5),
    "alpha_base_2": (-0.5, 1.5),
    "alpha_base_3": (-0.5, 1.5),
    "alpha_ee_1": (-0.5, 1.5),
    "alpha_ee_2": (-0.5, 1.5),
    "alpha_ee_3": (-0.5, 1.5),
    "beta_base_1": (-0.5, 0.5),
    "beta_base_2": (-0.5, 0.5),
    "beta_base_3": (-0.5, 0.5),
    "beta_base_4": (-0.5, 0.5),
    "beta_base_5": (-0.5, 0.5),
    "beta_base_6": (-0.5, 0.5),
    "beta_ee_1": (-0.5, 0.5),
    "beta_ee_2": (-0.5, 0.5),
    "beta_ee_3": (-0.5, 0.5),
    "beta_ee_4": (-0.5, 0.5),
    "beta_ee_5": (-0.5, 0.5),
    "beta_ee_6": (-0.5, 0.5),
    "gamma_base_1": (0.5, 2.0),
    "gamma_base_2": (0.5, 2.0),
    "gamma_base_3": (0.5, 2.0),
    "gamma_ee_1": (0.5, 2.0),
    "gamma_ee_2": (0.5, 2.0),
    "gamma_ee_3": (0.5, 2.0),
}

################################ Normalization Stats ################################
# using all datasets

SPOT_MODEL_NORMALIZATION_STATS: Dict = {
    "x_mean": jnp.array(
        [
            -0.54,
            -0.017,
            0.103,
            0.003,
            0.017,
            -0.007,
            0.384,
            -0.056,
            0.697,
            -0.0,
            0.018,
            -0.013,
            0.01,
            0.014,
            0.002,
            0.001,
            0.013,
            -0.003,
        ]
    ),
    "x_std": jnp.array(
        [
            0.399,
            0.227,
            0.282,
            0.223,
            0.195,
            0.237,
            0.387,
            0.318,
            0.196,
            0.271,
            0.308,
            0.202,
            0.223,
            0.223,
            0.256,
            0.166,
            0.12,
            0.197,
        ]
    ),
    "y_mean": jnp.array(
        [
            -0.539,
            -0.016,
            0.102,
            0.002,
            0.016,
            -0.007,
            0.383,
            -0.055,
            0.697,
            -0.001,
            0.017,
            -0.014,
        ]
    ),
    "y_std": jnp.array(
        [
            0.4,
            0.226,
            0.283,
            0.223,
            0.194,
            0.237,
            0.386,
            0.318,
            0.196,
            0.271,
            0.308,
            0.201,
        ]
    ),
}

SPOT_MODEL_NORMALIZATION_STATS_ENCODED_ANGLE: Dict = {
    "x_mean": jnp.array(
        [
            -0.54,
            -0.017,
            0.094,
            0.957,
            0.003,
            0.017,
            -0.007,
            0.384,
            -0.056,
            0.697,
            -0.0,
            0.018,
            -0.013,
            0.01,
            0.014,
            0.002,
            0.001,
            0.013,
            -0.003,
        ]
    ),
    "x_std": jnp.array(
        [
            0.399,
            0.227,
            0.258,
            0.095,
            0.223,
            0.195,
            0.237,
            0.387,
            0.318,
            0.196,
            0.271,
            0.308,
            0.202,
            0.223,
            0.223,
            0.256,
            0.166,
            0.12,
            0.197,
        ]
    ),
    "y_mean": jnp.array(
        [
            -0.539,
            -0.016,
            0.093,
            0.957,
            0.002,
            0.016,
            -0.007,
            0.383,
            -0.055,
            0.697,
            -0.001,
            0.017,
            -0.014,
        ]
    ),
    "y_std": jnp.array(
        [
            0.4,
            0.226,
            0.258,
            0.095,
            0.223,
            0.194,
            0.237,
            0.386,
            0.318,
            0.196,
            0.271,
            0.308,
            0.201,
        ]
    ),
}
