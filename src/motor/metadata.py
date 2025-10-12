"""
Metadata definitions for gait preprocessing:
- Feature groups per modality
- Risk thresholds
- Identifier columns
"""


IDENTIFIER_COLS = ['VISNO', 'patno', 'subject_id']


AXIVITY_FEATURE_GROUPS = {
    "data_quality": ['numberofdays', 'validdays6hr', 'validdays12hr',
                     'nonweardetected', 'upsidedowndetected'],
    "time_percent": [],  # filled dynamically by matching 'percent' or 'time'
    "svm": [],           # filled dynamically by matching 'svm'
    "step_bout": [],     # filled dynamically by matching 'step', 'bout', 'nap'
    "rotation": [],      # filled dynamically by 'rotation'
    "gait": [],          # cadence, rms, amp, stpreg, stepasym
    "variability": [],   # cv, sampentropy
    "std": []            # columns ending with 'std'
}


OPALS_FEATURE_GROUPS = {
    'Walking Speed and Cadence': ['SP_U', 'CAD_U'],
    'Arm Swing Amplitude and Variability': ['RA_AMP_U', 'LA_AMP_U',
                                            'RA_STD_U', 'LA_STD_U'],
    'Symmetry and Asymmetry Measures': ['SYM_U', 'ASYM_IND_U'],
    'Stride and Step Timing and Regularity': ['STR_T_U', 'STR_CV_U',
                                              'STEP_REG_U', 'STEP_SYM_U'],
    'Movement Smoothness / Jerk Measures': ['JERK_T_U', 'R_JERK_U', 'L_JERK_U'],
    'Functional Mobility (TUG) Test Metrics': ['TUG1_DUR', 'TUG1_STEP_NUM',
                                               'TUG1_STRAIGHT_DUR', 'TUG1_TURNS_DUR',
                                               'TUG1_STEP_REG', 'TUG1_STEP_SYM']
}

AXIVITY_THRESHOLDS = {
    'gait': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': False},
    'variability': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': False},
    'svm': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': True},
    'step_bout': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': False},
}

OPALS_THRESHOLDS = {
    'Walking Speed and Cadence': {'High Risk': 1.04, 'Moderate Risk': 1.19, 'invert': True},
    'Arm Swing Amplitude and Variability': {'High Risk': 15, 'Moderate Risk': 30, 'invert': False},
    'Symmetry and Asymmetry Measures': {'High Risk': 0.20, 'Moderate Risk': 0.15, 'invert': False},
    'Stride and Step Timing and Regularity': {'High Risk': 10, 'Moderate Risk': 7, 'invert': False},
    'Movement Smoothness / Jerk Measures': {'High Risk': 0.35, 'Moderate Risk': 0.20, 'invert': False},
    'Functional Mobility (TUG) Test Metrics': {'High Risk': 14.5, 'Moderate Risk': 12, 'invert': False}
}
