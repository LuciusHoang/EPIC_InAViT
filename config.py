import os

# 🔗 Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths to data
TRAIN_DATA_DIR = "/Volumes/T7 Shield/train/"
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test/")

# 🔎 Tasks List
SELECTED_TASKS = [
    "add_remove_lid",
    "basic_pick_place",
    "insert_remove_usb",
    "insert_remove_drawer",
    "insert_remove_tennis_ball",
    "insert_remove_plug_socket",
    "pick_place_food",
    "open_close_insert_remove_box",
    "stack_unstack_cups",
    "stack_unstack_bowls",
    "stack_unstack_plates",
    "stack_remove_jenga",
    "sort_beads",
    "scoop_dump_ice",
    "screw_unscrew_bottle_cap",
    "screw_unscrew_allen_fixture",
    "thread_unthread_bead_necklace",
    "clip_unclip_papers",
    "staple_paper",
    "tie_and_untie_shoelace",
    "fold_stack_unstack_unfold_cloths",
    "fold_unfold_paper_basic",
    "peel_place_sticker",
    "wrap_unwrap_food",
    "clean_tableware",
    "wipe_kitchen_surfaces",
    "wash_kitchen_dishes",
    "dry_hands",
    "make_sandwich"
]

# 🛠️ Model Hyperparameters
SEQ_LENGTH = 60          # number of frames per sequence
FRAME_SIZE = (224, 224)  # video frame size
FPS = 10                 # frames per second
NUM_CLASSES = len(SELECTED_TASKS)

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
SEED = 42

# 💾 Logging and Checkpoint
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints/")

# 🚀 Hardware Configuration
USE_MPS = True  # Set to True to enable Apple MPS acceleration
NUM_WORKERS = 0  # Number of workers for DataLoader

# Optional: Pre-trained models (if using transfer learning)
# PRETRAINED_RESNET = "resnet18"  # Change if using a different backbone

# Optional: Experiment Tag (for logging)
EXPERIMENT_NAME = "egodex_classification_baseline"
