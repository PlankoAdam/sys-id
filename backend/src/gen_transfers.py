import numpy as np
from control import TransferFunction, step_response, forced_response
import os
import time

from src.core.types import TFType
from src.core.utils import *

# ----------------------------
# CONFIGURATION
# ----------------------------
from src.core.config import *

# RANDOM_SEED = 42
RANDOM_SEED = int(time.time())
np.random.seed(RANDOM_SEED)
# np.random.seed(int(time.time()))

# ----------------------------
# SYSTEM GENERATION
# ----------------------------

# -------------------------------
# First order
# -------------------------------
def rand_o1_tf() -> tuple[TransferFunction, np.ndarray]:
    """
    Generate a random 1st-order stable transfer function.
    Returns:
        - TransferFunction object
        - Parameters: [K, T]
    """
    K = np.random.uniform(O1_K_MIN, O1_K_MAX)    
    T = np.random.uniform(O1_T_MIN, O1_T_MAX)
    
    tf = o1_tf_from_params(K, T)
    params = np.array([K, T], dtype=float)
    return tf, params

def rand_o1astat_tf() -> tuple[TransferFunction, np.ndarray]:
    """
    Generate a random 1st-order astatic stable transfer function.
    Returns:
        - TransferFunction object
        - Parameters: [K]
    """
    K = np.random.uniform(O1ASTAT_K_MIN, O1ASTAT_K_MAX)
    
    tf = o1astat_tf_from_params(K)
    params = np.array([K], dtype=float)
    return tf, params

# -------------------------------
# Second order aperiodic
# -------------------------------
def rand_o2aper_tf() -> tuple[TransferFunction, np.ndarray]:
    """
    Generate a random 2nd-order aperiodic stable transfer function.
    Returns:
        - TransferFunction object
        - Parameters: [K, T1, T2] in canonical form
    """
    K =  np.random.uniform(O2APER_K_MIN,  O2APER_K_MAX)    
    T1 = np.random.uniform(O2APER_T1_MIN, O2APER_T1_MAX)
    T2 = np.random.uniform(O2APER_T2_MIN, O2APER_T2_MAX)

    tf = o2aper_tf_from_params(K, T1, T2)
    params = np.array([K, T1, T2], dtype=float)
    return tf, params

# -------------------------------
# Second order periodic
# -------------------------------
def rand_o2per_tf() -> tuple[TransferFunction, np.ndarray]:
    """
    Generate a random 2nd-order periodic stable transfer function.
    Returns:
        - TransferFunction object
        - Parameters: [K, T, b] in canonical form
    """
    K = np.random.uniform(O2PER_K_MIN, O2PER_K_MAX)
    T = np.random.uniform(O2PER_T_MIN, O2PER_T_MAX)
    b = np.random.uniform(O2PER_B_MIN, O2PER_B_MAX)
    
    tf = o2per_tf_from_params(K, T, b)
    params = np.array([K, T, b], dtype=float)
    return tf, params

def rand_o2astat_tf() -> tuple[TransferFunction, np.ndarray]:
    """
    Generate a random 2nd-order astatic stable transfer function.
    Returns:
        - TransferFunction object
        - Parameters: [K, T]
    """
    K = np.random.uniform(O2ASTAT_K_MIN, O2ASTAT_K_MAX)    
    
    tf = o2astat_tf_from_params(K)
    params = np.array([K], dtype=float)
    return tf, params

def rand_o2astat_t_tf() -> tuple[TransferFunction, np.ndarray]:
    """
    Generate a random 2nd-order astatic stable transfer function with time constant.
    Returns:
        - TransferFunction object
        - Parameters: [K, T]
    """
    K = np.random.uniform(O2ASTATT_K_MIN, O2ASTATT_K_MAX)    
    T = np.random.uniform(O2ASTATT_T_MIN, O2ASTATT_T_MAX)
    
    tf = o2astat_t_tf_from_params(K, T)
    params = np.array([K, T], dtype=float)
    return tf, params

def rand_stable_tf(type: TFType | None = None):
    """
    Generate a random stable continuous-time transfer function.
    Can include oscillatory (complex-conjugate) pole pairs.
    """
    r = type or TFType.rand()

    if r == TFType.ORD1:
        return rand_o1_tf()
    elif r == TFType.ORD1_ASTAT:
        return rand_o1astat_tf()
    elif r == TFType.ORD2_APER:
        return rand_o2aper_tf()
    elif r == TFType.ORD2_PER:
        return rand_o2per_tf()
    elif r == TFType.ORD2_ASTAT:
        return rand_o2astat_tf()
    elif r == TFType.ORD2_ASTAT_T:
        return rand_o2astat_t_tf()
    
# ----------------------------
# SIMULATION FUNCTIONS
# ----------------------------

def simulate_step(tf: TransferFunction) -> tuple[np.ndarray, float]:
    """
    Simulate the step response of a transfer function.
    Args:
        tf: TransferFunction object
    Returns:
        tuple: (step response array, end time of the simulation)
    """
    time_end = np.random.uniform(STEP_TIME_END_MIN, STEP_TIME_END_MAX)
    t = np.linspace(STEP_TIME_START, time_end, TIME_POINTS)
    _, y = step_response(tf, T=t)
    return y, time_end

def simulate_step_osc(tf: TransferFunction) -> tuple[np.ndarray, float]:
    """
    Simulate the oscillating (square) step response of a transfer function.
    Args:
        tf: TransferFunction object
    Returns:
        tuple: (step response array, end time of the simulation)
    """
    t_mult = 4
    time_end = np.random.uniform(STEP_TIME_END_MIN, STEP_TIME_END_MAX) * t_mult
    t = np.linspace(STEP_TIME_START, time_end, TIME_POINTS)

    sec_sz = int(TIME_POINTS/4)
    inp = np.concat([np.full(sec_sz,1),np.full(sec_sz,0),np.full(sec_sz,1),np.full(sec_sz,0),])

    res = forced_response(tf, t, inp)
    return res.outputs, time_end

def add_noise(signal: np.ndarray, noise_std: float) -> np.ndarray:
    noise = np.random.random(size=signal.shape)
    noise = (noise - 0.5) * signal * noise_std
    return signal + noise

def gaussian_kernel(size, sigma):
    x = np.arange(-size//2 + 1, size//2 + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize
    return kernel

def smooth(signal: np.ndarray) -> np.ndarray:
    k_sz = int(TIME_POINTS/15)
    k = gaussian_kernel(size = k_sz, sigma = 1.0)

    beg = np.full(k_sz, signal[0])
    end = np.full(k_sz, signal[-1])
    s = np.concat([beg, signal, end])

    ret = np.convolve(s, k, mode='same')
    ret = ret[int(k_sz):len(ret)-int(k_sz)]
    return ret

# ----------------------------
# DATASET GENERATION
# ----------------------------
def generate_dataset(num_samples: int, tf_type: TFType, parent_dir: str, osc: bool = False):
    X_step = []
    X_Tend = []
    Y_params = []
    
    print(f"Generating dataset of {tf_type.name} transfer functions...")

    for _ in range(num_samples):
        
        tf, tf_params = rand_stable_tf(tf_type)
            
        if not tf:
            continue

        # Step response
        step, time_end = simulate_step_osc(tf) if osc else simulate_step(tf) 

        # Save response with 3 levels of noise
        for mult in [0.0, 0.5, 1.0]:
            step_noise = add_noise(step, NOISE_STD * mult)
            step_smooth = smooth(step_noise)

            # Inputs
            X_step.append(step_smooth)

            # Labels (tf parameters)
            Y_params.append(tf_params)
            X_Tend.append(time_end)
        
    print("Dataset generated!")
    
    save_X_step = np.array(X_step)
    save_X_Tend = np.array(X_Tend)
    save_Y_params = np.array(Y_params)
    
    dir = os.path.join(parent_dir, tf_type.name)
    
    print(f"Saving dataset to {dir}")
    
    try:
        os.mkdir(parent_dir)
    except FileExistsError:
        pass
    
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    
    np.save(os.path.join(dir, "X_step.npy"), save_X_step)
    np.save(os.path.join(dir, "X_Tend.npy"), save_X_Tend)
    np.save(os.path.join(dir, "Y_params.npy"), save_Y_params)
    
    print("Saved: X_step.npy, X_Tend.npy, Y_params.npy")
