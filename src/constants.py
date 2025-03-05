KAGGLE_TRAIN = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
KAGGLE_TEST = '/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
KAGGLE_DESCRIPTION = '/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'
LOCAL_TRAIN = 'train.csv'
LOCAL_TEST = 'test.csv'
LOCAL_DESCRIPTION = 'data_dictionary.csv'
RMV = ["ID","efs","efs_time","y"]
FOLDS = 10

HLA_GEN_EXPRS = [
    ('diff', 'hla_match_c_low', 'hla_match_drb1_low'),
    ('diff', 'hla_match_c_low', 'hla_match_drb1_high'),
    ('diff', 'hla_match_drb1_low', 'hla_match_a_high'),
    ('diff', 'hla_match_drb1_low', 'hla_match_a_low'),
    ('diff', 'hla_match_drb1_low', 'hla_match_b_high'),
    ('diff', 'hla_match_a_high', 'hla_match_drb1_high'),
    ('diff', 'hla_match_a_low', 'hla_match_drb1_high'),
    ('diff', 'hla_match_b_high', 'hla_match_drb1_high')
]