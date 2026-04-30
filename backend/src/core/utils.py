import numpy as np
from control import TransferFunction
import torch

from src.core.config import *
from src.core.types import TFType

def load_csv(path: str) -> np.ndarray:
    """
    Loads a CSV file into a NumPy matrix.
    Assumes the CSV has no header and contains only numbers.
    """
    try:
        matrix = np.loadtxt(path, delimiter=',', dtype=np.float32)
        print(matrix)
        shapes = [2,3]
        if len(matrix.shape) in shapes and matrix.shape[1] in shapes:
            matrix = matrix.T  # Swaps rows and columns
        return matrix
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
def strip_leading_zeros(time_points: np.array, step_resposne: np.array, input_func: np.array = None, threshold: float = 1e-6) -> tuple[np.array, np.array]:
    """
    Strips leading zeros from a step response array, and adjust the time points accordingly.
    """
    nonzero_indices = np.where(np.abs(step_resposne) > threshold)[0]

    if len(nonzero_indices) <= 0:
        return [], [], []
    
    first_nonzero = max(0, nonzero_indices[0] - 1)

    return time_points[first_nonzero:], step_resposne[first_nonzero:], (input_func[first_nonzero:] if input_func else None)

def find_first_non_zero_idx(arr: np.array, threshold: float = 1e-6) -> int:
    nonzero_indices = np.where(np.abs(arr) > threshold)[0]

    if len(nonzero_indices) == 0:
        return 0
    
    first_nonzero = nonzero_indices[0]
    return first_nonzero

def normalize_step_time(Y, norm_end_time, norm_start_time=0.0, time_points=TIME_POINTS):
    """
    Normalize the time axis of a step response to a fixed range.

    :param Y: Step response data
    :param end_time: End time of the original step response

    :return: Tuple of (normalized time array, normalized step response array)
    """
    y = np.array(Y)

    t = np.linspace(norm_start_time, norm_end_time, y.shape[0])
    t_norm = np.linspace(norm_start_time, norm_end_time, time_points)
    y_norm = np.interp(t_norm, t, y)

    return np.array(t_norm), np.array(y_norm)

def load_and_process_csv(path: str, threshold: float = 1e-6) -> tuple[np.array, np.array]:
    m = load_csv(path)

    t = np.linspace(0.0, np.max(m[0]), len(m[0]))


    # import matplotlib.pyplot as plt
    # print(t)
    # print(m[0])
    # plt.plot(t)
    # plt.plot(m[0])
    # plt.show()

    t_strip, sr_strip, _ = strip_leading_zeros(t, m[1] if m.shape[0] == 2 else m[2], threshold=threshold)
    t_norm, sr_norm = normalize_step_time(sr_strip, norm_end_time=t_strip[-1])

    print(sr_norm)
    print(np.max(m[0]))
    print(len(m[1]))

    return t_norm, sr_norm

def tf_from_params(tftype: TFType, params: list[float]) -> TransferFunction:
    if tftype == TFType.ORD1:
        tf = o1_tf_from_params(K=params[0], T=params[1])
    elif tftype == TFType.ORD1_ASTAT:
        tf = o1astat_tf_from_params(K=params[0])
    elif tftype == TFType.ORD2_APER:
        tf = o2aper_tf_from_params(K=params[0], T1=params[1], T2=params[2])
    elif tftype == TFType.ORD2_PER:
        tf = o2per_tf_from_params(K=params[0], T=params[1], b=params[2])
    elif tftype == TFType.ORD2_ASTAT:
        tf = o2astat_tf_from_params(K=params[0])
    elif tftype == TFType.ORD2_ASTAT_T:
        tf = o2astat_t_tf_from_params(K=params[0], T=params[1])

    return tf

def o1_tf_from_params(K: float, T: float) -> TransferFunction:
    """
    Create a 1st-order transfer function from parameters.
    Args:
        K: Gain
        T: Time constant
    Returns:
        TransferFunction object
    """
    num = [K]
    den = [T, 1]
    tf = TransferFunction(num, den)
    return tf

def o1astat_tf_from_params(K: float) -> TransferFunction:
    """
    Create a 1st-order astatic transfer function from parameters.
    Args:
        K: Gain
    Returns:
        TransferFunction object
    """
    num = [K]
    den = [1, 0]
    tf = TransferFunction(num, den)
    return tf

def o2aper_tf_from_params(K: float, T1: float, T2: float) -> TransferFunction:
    """
    Create a 2nd-order aperiodic transfer function from parameters.
    Args:
        K: Gain
        T1: Time constant 1
        T2: Time constant 2
    Returns:
        TransferFunction object
    """
    num = [K]
    den = [T1*T2, T1+T2, 1]
    tf = TransferFunction(num, den)
    return tf

def o2per_tf_from_params(K: float, T: float, b: float) -> TransferFunction:
    """
    Create a 2nd-order periodic transfer function from parameters.
    Args:
        K: Gain
        T: Time constant
        b: Damping ratio
    Returns:
        TransferFunction object
    """
    num = [K]
    den = [T**2, 2*b*T, 1]
    tf = TransferFunction(num, den)
    return tf

def o2astat_tf_from_params(K: float) -> TransferFunction:
    """
    Create a 2nd-order astatic transfer function from parameters.
    Args:
        K: Gain
    Returns:
        TransferFunction object
    """
    num = [K]
    den = [1, 0, 0]
    tf = TransferFunction(num, den)
    return tf

def o2astat_t_tf_from_params(K: float, T: float) -> TransferFunction:
    """
    Create a 2nd-order astatic transfer function with time constant from parameters.
    Args:
        K: Gain
        T: Time constant
    Returns:
        TransferFunction object
    """
    num = [K]
    den = [T, 1, 0]
    tf = TransferFunction(num, den)
    return tf

################################################
# Step response functions for fast computation #
################################################

# Book equation no. 5.4
def step_response_o1(params, t):
    K = params[:,0:1]
    T1 = params[:,1:2]
    # t = t.unsqueeze(0)

    exp_term = torch.exp(-t/T1)

    y = K * (1 - exp_term)
    return y

# Book equation no. 5.42
def step_response_o1astat(params, t):
    K = params[:,0:1]
    # t = t.unsqueeze(0)

    y = K*t
    return y

# Book equation no. 5.13
def step_response_o2aper(params, t):
    K = params[:,0:1]
    T1 = params[:,1:2]
    T2 = params[:,2:3]
    # t = t.unsqueeze(0)

    # term_1 = (T1*torch.exp(-t/torch.clamp(T1, min=1e-6))) / torch.clamp(T1-T2, min=1e-6)
    # term_2 = (T2*torch.exp(-t/torch.clamp(T2, min=1e-6))) / torch.clamp(T1-T2, min=1e-6)
    term_1 = (T1*torch.exp(-t/torch.clamp(T1, min=1e-6))) / T1-T2
    term_2 = (T2*torch.exp(-t/torch.clamp(T2, min=1e-6))) / T1-T2

    y = K * (1-term_1+term_2)
    return y

# Book equation no. 5.25
def step_response_o2per(params, t):
    K = params[:,0:1]
    T1 = params[:,1:2]
    b = params[:,2:3]
    # t = t.unsqueeze(0)

    T1 = torch.clamp(T1, min=1e-6)
    wn = 1.0 / T1
    wd = wn * torch.sqrt(1 - b**2)
    phi = torch.atan(torch.sqrt(1 - b**2) / b)

    exp_term = torch.exp(-b * wn * t)
    sin_term = torch.sin(wd * t + phi)

    y = K * (1 - (1/torch.sqrt(1-b**2)) * exp_term * sin_term)
    # print(f"o2per output: {y}")
    return y

def step_response_o2astat(params, t):
    K = params[:,0:1]
    # t = t.unsqueeze(0)

    y = K*torch.square(t)/2
    # print(f"o2astat output: {y}")
    return y

# Book equation no. 5.46
def step_response_o2astat_t(params, t):
    K = params[:,0:1]
    T1 = params[:,1:2]
    # t = t.unsqueeze(0)

    T1 = torch.clamp(T1, min=1e-6)
    exp_term = T1 * torch.exp(-t/T1)

    y = K*(t-T1+exp_term)
    # print(f"o2astat_t output: {y}")
    return y

def step_response_of_type(tftype: TFType, params, t):
    if tftype == TFType.ORD1:
        return step_response_o1(params, t)
    if tftype == TFType.ORD1_ASTAT:
        return step_response_o1astat(params, t)
    if tftype == TFType.ORD2_APER:
        return step_response_o2aper(params, t)
    if tftype == TFType.ORD2_PER:
        return step_response_o2per(params, t)
    if tftype == TFType.ORD2_ASTAT:
        return step_response_o2astat(params, t)
    if tftype == TFType.ORD2_ASTAT_T:
        return step_response_o2astat_t(params, t)