import torch

from src.tftype_classifier.classes import *
from src.core.types import TFType
from src.core.config import *

def load_model(input_dim: int = TIME_POINTS, osc: bool = False):
    parent_dir = OSC_DIR if osc else STEP_DIR
    model = TFTypeClassifier(input_dim=input_dim, num_classes=len(TFType))
    model.load_state_dict(torch.load(f"{parent_dir}/{MODELS_DIR}/tftype_classifier.pt", map_location=torch.device('cpu')))
    return model

def predict_tftype(step_response, osc:bool = False):
    model = load_model(input_dim=len(step_response), osc=osc)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(step_response, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        return TFType(logits.argmax(dim=1).item())
    
def predict_tftype_with_model(step_response, model):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(step_response, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        return TFType(logits.argmax(dim=1).item())
