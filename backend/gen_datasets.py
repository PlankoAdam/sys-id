from src.gen_transfers import generate_dataset
from src.core.config import *
from src.core.types import TFType
    
def gen_all(osc: bool):
    parent_dir = OSC_DIR if osc else STEP_DIR
    print(f"Generating datasets: osc={osc}, parent_dir={parent_dir}")
    for tftype in TFType:
        # Training data
        generate_dataset(TRAIN_NUM_SAMPLES, tftype, f"{parent_dir}/{TRAIN_SAVE_DIR}", osc=osc)
        # Validation data
        generate_dataset(VAL_NUM_SAMPLES, tftype, f"{parent_dir}/{VAL_SAVE_DIR}", osc=osc)

if __name__ == "__main__":
    gen_all(False)
    gen_all(True)