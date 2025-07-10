import os

# 🔗 Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
TRAIN_DATA_DIR = os.environ.get("TRAIN_DATA_DIR", "/Volumes/T7_Shield/train/")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test/")

# 🔎 Task List
SELECTED_TASKS = [
    "add_remove_lid", "basic_pick_place", "clean_tableware", "clip_unclip_papers",
    "dry_hands", "fold_stack_unstack_unfold_cloths", "fold_unfold_paper_basic",
    "insert_remove_drawer", "insert_remove_plug_socket", "insert_remove_tennis_ball",
    "insert_remove_usb", "make_sandwich", "open_close_insert_remove_box",
    "peel_place_sticker", "pick_place_food", "scoop_dump_ice",
    "screw_unscrew_allen_fixture", "screw_unscrew_bottle_cap", "sort_beads",
    "stack_remove_jenga", "stack_unstack_bowls", "stack_unstack_cups",
    "stack_unstack_plates", "staple_paper", "thread_unthread_bead_necklace",
    "tie_and_untie_shoelace", "vertical_pick_place", "wash_kitchen_dishes",
    "wipe_kitchen_surfaces", "wrap_unwrap_food"
]

# 🛠️ Model + Data Params
SEQ_LENGTH = 60
FRAME_SIZE = (224, 224)
FPS = 10
NUM_CLASSES = len(SELECTED_TASKS)
INPUT_DIM = 36

# 🏋️ Training Settings
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
SEED = 42

# 💾 Logging
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints/")

# 🚀 Hardware (CPU only)
DEVICE = "cpu"
NUM_WORKERS = 0

# 🧠 Model Variants
MODEL_TYPES = ['mlp', 'cnn_lstm', 'transformer']

# 📌 Experiment
EXPERIMENT_NAME = "egodex_classification_baseline"

class Config:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.train_data_path = TRAIN_DATA_DIR
        self.test_data_path = TEST_DATA_DIR

        self.task_list = SELECTED_TASKS
        self.num_classes = NUM_CLASSES

        self.seq_length = SEQ_LENGTH
        self.frame_size = FRAME_SIZE
        self.fps = FPS
        self.input_dim = INPUT_DIM

        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.seed = SEED

        self.log_dir = LOG_DIR
        self.checkpoint_dir = CHECKPOINT_DIR
        self.experiment_name = EXPERIMENT_NAME

        self.device = DEVICE
        self.num_workers = NUM_WORKERS

        self.model_types = MODEL_TYPES

    def __repr__(self):
        return f"<Config: {self.experiment_name} | device={self.device}, tasks={self.num_classes}>"

if __name__ == "__main__":
    cfg = Config()
    print(cfg)
