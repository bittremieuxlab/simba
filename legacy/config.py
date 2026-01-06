import os


class Config:
    # default configuration
    # Spectra and spectrum pairs to include with the following settings.
    def __init__(self):
        # device
        self.ACCELERATOR = "gpu"
        # MULTITASKING
        self.COLUMN_EDIT_DISTANCE = 2
        self.COLUMN_MCES20 = 3

        # PREPROCESSING
        self.PREPROCESSING_BATCH_SIZE = 1000
        self.PREPROCESSING_NUM_WORKERS = 60
        self.PREPROCESSING_NUM_NODES = 10
        # current node for preprocessing
        self.PREPROCESSING_CURRENT_NODE = None
        # overwrite the output file during generation
        self.PREPROCESSING_OVERWRITE = False
        # path to the spectra file
        self.SPECTRA_PATH = None
        # maximum number of spectra to use for training
        self.MAX_SPECTRA_TRAIN = 1000
        # maximum number of spectra to use for validation
        self.MAX_SPECTRA_VAL = 1000
        # maximum number of spectra to use for testing
        self.MAX_SPECTRA_TEST = 1000
        # fraction of spectra to use for validation
        self.VAL_SPLIT = 0.1
        # fraction of spectra to use for testing
        self.TEST_SPLIT = 0.1
        # self.COMPUTE_SPECIFIC_PAIRS=True
        self.USE_LEARNABLE_MULTITASK = True
        self.FORMAT_FILE_SPECIFIC_PAIRS = "INPUT_SPECIFIC_PAIRS_indexes_tani_incremental"  # the prefix of the file containing the indexes to be computed
        # self.USE_EDIT_DISTANCE=False ## If using edit distance for generating data, not for training!!!
        self.SUBSAMPLE_PREPROCESSING = False
        self.RANDOM_MCES_SAMPLING = False
        self.CHARGES = [1]
        self.MIN_N_PEAKS = 6
        self.FRAGMENT_MZ_TOLERANCE = 0.1
        self.MIN_MASS_DIFF = 0  # Da
        self.MAX_MASS_DIFF = 200  # Da
        self.THRESHOLD_MCES = 20
        self.USE_PRECURSOR_MZ_FOR_MODEL = True

        # Metadata
        self.USE_ONLY_PROTONIZED_ADDUCTS = False
        self.USE_ADDUCT = False
        # Input adduct info as categorical variables
        self.CATEGORICAL_ADDUCTS = False
        self.ADDUCT_MASS_MAP_CSV = None
        self.USE_CE = False
        self.USE_ION_ACTIVATION = False
        self.USE_ION_METHOD = False

        ## FOR COMPUTING EDIT DISTANCE LOCALLY
        self.USE_FINGERPRINT = False
        self.USE_EDIT_DISTANCE = (
            True  ## If using edit distance for generating data, not for training!!!
        )
        self.COMPUTE_SPECIFIC_PAIRS = False

        # training
        self.TRAINING_NUM_WORKERS = 10
        self.USE_RESAMPLING = False
        self.TRANSFORMER_CONTEXT = 100  ##number of input peaks to the transformer
        self.ADD_HIGH_SIMILARITY_PAIRS = False
        self.USE_MOLECULAR_FINGERPRINTS = False
        self.USE_MCES20_LOG_LOSS = False  ### apply log function to increase the weight of the differences in the low range
        self.USE_EDIT_DISTANCE_REGRESSION = False
        self.USE_MULTITASK = True
        self.EDIT_DISTANCE_N_CLASSES = 6
        self.EDIT_DISTANCE_USE_GUMBEL = False
        self.TAU_GUMBEL_SOFTMAX = 10
        self.GUMBEL_REG_WEIGHT = 0.1
        self.USE_TANIMOTO = False  # using Tanimoto or MCES20 for training
        self.EDIT_DISTANCE_MAX_VALUE = 666
        self.MCES20_MAX_VALUE = 40  # value used as midpoint for normalization. 19 it is chosen to make NORMALIZED_MCES to be in the range below 0.49 and make it consider a low similarity pair
        self.USE_LOSS_WEIGHTS_SECOND_SIMILARITY = (
            False  # use weights for training the second similarity of multitasking
        )
        self.N_LAYERS = 5  # transformer parameters
        self.D_MODEL = 256  # transformer parameters
        self.EMBEDDING_DIM = 512
        self.use_cosine_distance = True
        self.LR = 1e-4
        self.epochs = 1000
        self.VAL_CHECK_INTERVAL = 10000
        self.BATCH_SIZE = 128
        self.enable_progress_bar = True
        self.threshold_class = 0.7  # threshold classification binary
        self.load_maldi_embedder = False
        self.INFERENCE_USE_LAST_MODEL = False
        self.maldi_embedder_path = None  # Set via --maldi_embedder_path
        self.load_pretrained = False  # a whole SIMBA model
        self.dataset_path = None  # Set via --dataset_path
        self.use_uniform_data_TRAINING = False
        self.bins_uniformise_TRAINING = 10
        self.use_uniform_data_INFERENCE = True
        self.bins_uniformise_INFERENCE = 10
        self.validate_after_ratio = 0.0010  # it indicates the interval between validations. O.1 means 10 validations in 1 epoch
        self.extra_info = "_multitasking_mces20raw_gumbelhard_20241004"

        self.PREPROCESSING_PICKLE_FILE = None
        self.PREPROCESSING_DIR = None
        self.PREPROCESSING_DIR_TRAIN = None
        self.PREPROCESSING_DIR_VAL_TEST = None
        self.MOL_SPEC_MAPPING_FILE = "edit_distance_neurips_nist_exhaustive.pkl"
        self.CHECKPOINT_DIR = None
        self.pretrained_path = None
        self.BEST_MODEL_NAME = "best_model.ckpt"
        self.PRETRAINED_MODEL_NAME = "pretrained_model.ckpt"
        self.derived_variables()

        ## TESTING
        self.UNIFORMIZE_DURING_TESTING = True

    def derived_variables(self):
        self.MODEL_CODE = f"{self.D_MODEL}_units_{self.N_LAYERS}_layers_{self.epochs}_epochs_{self.LR}_lr_{self.BATCH_SIZE}_bs{self.extra_info}"

        if self.CHECKPOINT_DIR is None:
            # Use environment variable or fallback to current working directory
            # Set CHECKPOINT_BASE env var to customize (e.g., for cluster: /scratch/antwerpen/209/vsc20939/data/model_checkpoints)
            checkpoint_base = os.environ.get("CHECKPOINT_BASE", "./checkpoints")
            self.CHECKPOINT_DIR = os.path.join(
                checkpoint_base, f"model_checkpoints_{self.MODEL_CODE}"
            )

        if self.pretrained_path is None:
            self.pretrained_path = os.path.join(
                self.CHECKPOINT_DIR, self.PRETRAINED_MODEL_NAME
            )
        self.best_model_path = os.path.join(self.CHECKPOINT_DIR, self.BEST_MODEL_NAME)
