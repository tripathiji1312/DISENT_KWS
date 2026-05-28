# config.py — THE CONTRACT. Do not change without both agreeing.
# All tensor shapes, hyperparameters, and dataset constants live here.

# ---------------------------------------------------------------------------
# Audio Front-end
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 16000
N_MELS          = 80
WIN_LENGTH      = 400       # 25 ms
HOP_LENGTH      = 160       # 10 ms
MAX_AUDIO_SEC   = 2.0
MAX_FRAMES      = 200       # ~2 s at 10 ms hop  →  (80, 200) feature tensor

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
EMBED_DIM          = 192
BC_CHANNELS        = [16, 16, 32, 48]   # progressive channel widths
TEMPORAL_CHANNELS  = 48

# Mamba SSM (if available)
MAMBA_D_STATE  = 16
MAMBA_D_CONV   = 4

# Causal Conformer (phonetic head)
CONFORMER_HEADS  = 4
CONFORMER_CONV_K = 15

# ECAPA-TDNN Lite (speaker head)
ECAPA_SCALE    = 4
ECAPA_SE_RATIO = 4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE             = 128
NUM_SPEAKERS_VOXCELEB  = 7205   # full VoxCeleb1+2
NUM_KEYWORDS_GSC       = 35     # Google Speech Commands v2

# AAM-Softmax
AAM_SCALE   = 30
AAM_MARGIN  = 0.2

# Prototypical loss
PROTO_SCALE  = 32
PROTO_MARGIN = 0.25

# Rejection (confuser) loss
REJECTION_MARGIN = 0.4

# Disentanglement
GRL_MAX_LAMBDA = 1.0
CLUB_WEIGHT    = 0.1

# Knowledge distillation
KD_TEMPERATURE = 4
KD_ALPHA       = 0.7

# ---------------------------------------------------------------------------
# Scoring / Detection
# ---------------------------------------------------------------------------
SCORE_W_KW  = 0.55   # weight on keyword similarity
SCORE_W_SPK = 0.45   # weight on speaker similarity
EMA_ALPHA   = 0.7    # streaming smoothing factor
DEFAULT_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
SPEC_FREQ_MASK = 15
SPEC_TIME_MASK = 25
SPEC_NUM_MASKS = 2
SNR_RANGE      = (-5, 30)    # dB
ROOM_DIM_RANGE = (3, 10)     # metres
RT60_RANGE     = (0.1, 1.0)  # seconds
DISTANCE_RANGE = (0.5, 5.0)  # metres
