import numpy as np
import time
import torch
import control as ct
import matplotlib.pyplot as plt
from collections import namedtuple

from src.tf_predictor.classes import *
from src.core.config import *
from src.core.utils import *
from src.core.types import TFType, PARAMS_NUM_MAP

def load_model(tftype: TFType, osc: bool = False) -> TFRegressor:
    MODEL_PATH = f"{OSC_DIR if osc else STEP_DIR}/{MODELS_DIR}/tf_predictor_{tftype.name}.pt"
    model = TFRegressor(input_dim=TIME_POINTS, output_dim=PARAMS_NUM_MAP.get(tftype), tftype=tftype)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return model

PredictionResults = namedtuple("PredictionResults", ['step_response', 'params', 'tf', 'input'])

def predict_with_model(model: TFRegressor, tftype: TFType, step_response: np.ndarray, time_end: float, osc:bool = False) -> tuple:
    """
    Predict the TF from the step response data, using the specified model.
    
    :param model: Model to use
    :type model: TFRegressor
    :param tftype: 
    :type tftype: TFType
    :param step_response: Description
    :type step_response: np.ndarray
    :param time_end: Description
    :type time_end: float
    
    :return: Tuple of (predicted_step_repsonse_array, predicted_params, predicted_TF)
    :rtype: tuple
    """
    with torch.no_grad():
        x = torch.tensor(step_response, dtype=torch.float32).unsqueeze(0)
        time_end_tensor = torch.tensor([time_end], dtype=torch.float32)
        params_pred = model(x, time_end_tensor).numpy().squeeze()
        params_pred = params_pred if params_pred.ndim > 0 else params_pred.reshape(1)

    tf_pred = tf_from_params(tftype=tftype, params=params_pred)
    
    t = np.linspace(0, int(time_end), len(step_response))

    if osc:
        sec_sz = int(len(step_response)/4)
        inp = np.concat([np.full(sec_sz,1),np.full(sec_sz,0),np.full(sec_sz,1),np.full(sec_sz,0)])
        resp = ct.forced_response(tf_pred, timepts=t, inputs=inp)
    else:
        resp = ct.step_response(tf_pred, T=t)

    return PredictionResults(resp.outputs, params_pred, tf_pred, resp.inputs)
