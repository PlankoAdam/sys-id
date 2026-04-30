from __future__ import annotations

from enum import Enum
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from src.core.config import *
# from src.core.utils import normalize_step_time

np.random.seed(int(time.time()))

class TFType(Enum):
    ORD1 = 0
    ORD1_ASTAT = 1
    ORD2_APER = 2
    ORD2_PER = 3
    ORD2_ASTAT = 4
    ORD2_ASTAT_T = 5
    
    def rand():
        r = np.random.randint(0, len(TFType))
        return TFType(r)
    
PARAMS_NUM_MAP: dict[TFType, int] = {
    TFType.ORD1: 2,
    TFType.ORD1_ASTAT: 1,
    TFType.ORD2_APER: 3,
    TFType.ORD2_PER: 3,
    TFType.ORD2_ASTAT: 1,
    TFType.ORD2_ASTAT_T: 2,
}

class System():
    def __init__(self, tftype: TFType, params: list[float], step_response: np.ndarray, time_end: float, idx: int = None):
        self.tftype = tftype
        self.params = params
        self.step_response = step_response
        self.time_end = time_end
        self.idx = idx
        
    def to_csv(self, path, include_time: bool = False):
        arr: np.ndarray = self.step_response
        
        if include_time:
            t = np.linspace(0, self.time_end, len(self.step_response))
            arr = np.asarray([t, self.step_response])
            arr = arr.transpose()
        
        np.savetxt(path, arr, delimiter=",")
                
    def plot(self):
        # t_norm, step_norm = normalize_step_time(self.step_response, norm_end_time=self.time_end, time_points=TIME_POINTS)

        # plt.plot(t_norm, step_norm)
        plt.plot(self.step_response)
        plt.title(f"System Type: {self.tftype.name} , Tmax={self.time_end}\nParameters: {self.params}")
        plt.xlabel("Time Steps")
        plt.ylabel("Response")
        plt.grid()
        plt.show()
        
    def get_random_sample(tftype: TFType = None, idx: int = None, osc: bool = False) -> System:
        """
        Get a random sample system.
        
        :return: A System object containing the TF type, parameters, and step response
        :rtype: System
        """
        # np.random.seed(int(time.time()))
        type:TFType = tftype if tftype is not None else TFType.rand()
        base_dir = f"{OSC_DIR if osc else STEP_DIR}/{VAL_SAVE_DIR}"
        
        step_response_path = os.path.join(base_dir, type.name, "X_step.npy")
        xs = np.load(step_response_path)

        Tend_path = os.path.join(base_dir, type.name, "X_Tend.npy")
        time_ends = np.load(Tend_path)
        
        params_path = os.path.join(base_dir, type.name, "Y_params.npy")
        params = np.load(params_path)
        
        idx = idx if idx is not None else np.random.randint(0, len(xs))
        return System(tftype=type, params=params[idx], step_response=xs[idx], time_end=time_ends[idx], idx=idx)
