import matplotlib.pyplot as plt
import numpy as np
# from control import step_response, TransferFunction, impulse_response
import control as ct
import torch

from src.core.utils import *
from src.core.types import *
from src.core.config import *
from src.gen_transfers import *
from src.tf_predictor.train_model import *

def main():
    K = 1.0
    
    # zetas = [0.1, 0.3, 0.5, 0.7, 0.9]
    # omega = 1.0
    
    b = 0.5
    Ts = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for T in Ts:
        tf = o2per_tf_from_params(K, T, b)
        t = np.linspace(0, 100, TIME_POINTS)
        _, sr = step_response(tf, T=t)
        
        # plt.plot(t, sr, label=f"ωn={omega}")
        plt.plot(t, sr, label=f"T={T}\nb={b}")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        
    plt.legend()
    plt.show()
    
def main4():
    K = 1.0
    
    bs = [0.1, 0.3, 0.5, 0.7, 0.9]
    T = 1.0
    
    # zeta = 0.5
    # omegas = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for b in bs:
        tf = o2per_tf_from_params(K, T, b)
        t = np.linspace(0, 100, TIME_POINTS)
        _, sr = step_response(tf, T=t)
        
        # plt.plot(t, sr, label=f"ωn={omega}")
        plt.plot(t, sr, label=f"T={T}\nb={b}")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        
    plt.legend()
    plt.show()
    
def main2():
    K = 1.0
    
    # T1s = [0.2, 0.5, 1.0, 2.0, 5.0]
    # T2 = 1.0
    
    T1 = 1.0
    T2s = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    for T2 in T2s:
        tf = o2aper_tf_from_params(K, T1, T2)
        t = np.linspace(0, 100, TIME_POINTS)
        _, sr = step_response(tf, T=t)
        
        plt.plot(t, sr, label=f"T1={T1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        
    plt.legend()
    plt.show()
    
def main3():
    K = 1.0
    tf1 = TransferFunction([0.5], [0.5, 0]) # K/s
    tf2 = TransferFunction([2], [2, 0]) # K/s^2
    t = np.linspace(0, 100, TIME_POINTS)
    _, sr1 = step_response(tf1, T=t)
    _, sr2 = step_response(tf2, T=t)
    
    plt.plot(t, sr1, label="K/s")
    plt.plot(t, sr2, label="K/s^2")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.show()
    
def main5():
    K = 1.0
    
    # T1s = [0.2, 0.5, 1.0, 2.0, 5.0]
    # T2 = 1.0
    
    Ts = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    for T in Ts:
        # tf = o1_tf_from_params(K, T)
        tf = TransferFunction([K], [1, T, 0])
        t = np.linspace(0, 20, TIME_POINTS)
        _, sr = step_response(tf, T=t)
        
        plt.plot(t, sr, label=f"T={T}")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        
    plt.legend()
    plt.show()

def main6():
    K = 1.0
    T = 1.0
    tf = TransferFunction([K], [1, 0, 0]) # K/s
    t = np.linspace(0, 20, TIME_POINTS)
    _, sr1 = step_response(tf, T=t)
    _, ir1 = impulse_response(tf, T=t)
    
    plt.plot(t, sr1, label="K/s")
    plt.plot(t, ir1, label="K/s")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.show()

def main7():
    K = 1.0
    b = 0.5
    T = 1.0
    tf = o2per_tf_from_params(K, T, b)
    t = np.linspace(0, 100, TIME_POINTS)
    _, sr1 = step_response(tf, T=t)
    _, ir1 = impulse_response(tf, T=t)
    
    plt.plot(t, sr1, label="K/s")
    plt.plot(t, ir1, label="K/s")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.show()

def main8():
    K = 1.0
    T = 1.0
    b = 0.2
    T = 1.0
    # tf = o1_tf_from_params(K, T)
    tf = o2per_tf_from_params(K, T, b)
    # tf = o2astat_tf_from_params(K, T)
    end_time = 10
    t = np.linspace(0, end_time, TIME_POINTS)
    _, sr = step_response(tf, T=t)
    # sr = add_noise(sr, noise_std=0.1)

    norm_times = [0.1*end_time, 0.5*end_time, end_time, 2*end_time, 4*end_time]

    for i, nt in enumerate(norm_times):
        norm_t, norm_y = normalize_step_time(sr, norm_end_time=nt, time_points=100)

        plt.subplot(1, len(norm_times), i+1)
        plt.plot(norm_t, norm_y)
        plt.title(f"Normalized to {nt}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")

    # plt.margins(0.1)
    # plt.tight_layout()
    plt.show()

def main9():
    m = load_csv(f"{DATA_DIR}/data1.csv")
    print(m.shape)

    t_strip, sr_strip, _ = strip_leading_zeros(m[0], m[1])
    print(f"Stripped shape: {t_strip.shape}, {sr_strip.shape}")

    print(f"start time: {t_strip[0]}")
    t_norm, sr_norm = normalize_step_time(sr_strip, norm_start_time=0.0, norm_end_time=t_strip[-1])
    print(f"Normalized shape: {t_norm.shape}, {sr_norm.shape}")

    plt.plot(t_norm, sr_norm, label="stripped, normalized data1.csv")
    plt.plot(t_strip, sr_strip, label="stripped data1.csv", linestyle="dashed")
    plt.plot(m[0], m[1], label="data1.csv", linestyle="dotted")
    plt.show()

def main10():
    t, sr = load_and_process_csv(f"{DATA_DIR}/data2.csv", threshold=0.01)

    plt.plot(t, sr, label="Processed data1.csv")
    plt.show()
    
def main11():
    tf = o1_tf_from_params(1,1)
    
    t = np.linspace(10, 20, TIME_POINTS)
    # inp = np.sin(t)
    # inp = np.linspace(0,1,TIME_POINTS)
    # inp = np.log(t)
    
    res = ct.forced_response(tf,t,1)
    
    # print(res.outputs)
    # plt.plot(t, inp, label="input")
    plt.plot(t, res.outputs, label="output")
    plt.show()
    
def main12():
    G = o1_tf_from_params(1,4)
    # G= o2per_tf_from_params(1,0.5,0.5)
    
    num_d, den_d = ct.pade(4, 5)
    d_tf = ct.tf(num_d, den_d)
    
    tf = G * d_tf
    
    t= np.linspace(0, 20, TIME_POINTS)
    # res = ct.forced_response(tf, t, 1)
    res = ct.forced_response(tf, t, 1)
    
    plt.plot(res.time, res.inputs)
    plt.plot(res.time, res.outputs)
    plt.show()

def save_rand():
    dir = "synth_data"
    
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    
    s = System.get_random_sample(TFType.ORD2_PER)
    s.to_csv(os.path.join(dir, f"{s.tftype.name}-{s.idx}.csv"), include_time=True)
    
def save_n_rand(n: int, osc: bool = False):
    dir = f"{OSC_DIR if osc else STEP_DIR}/synth_data"
    
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    
    for i in range(n):
        s = System.get_random_sample()
        s.to_csv(os.path.join(dir, f"{s.tftype.name}-{s.idx}.csv"), include_time=True)

def main13():
    arr = np.genfromtxt("real_data/data1i.csv", delimiter=',')
    arr = arr.transpose()

    G = o1_tf_from_params(0.9,1)
    # G= o2per_tf_from_params(1,0.5,0.5)
    
    num_d, den_d = ct.pade(0, 5)
    d_tf = ct.tf(num_d, den_d)
    
    tf = G * d_tf
    
    # t= np.linspace(0, 20, TIME_POINTS)
    # res = ct.forced_response(tf, t, 1)

    t= arr[0]
    res = ct.forced_response(tf, t, arr[2])
    
    plt.plot(res.time, res.inputs, label="in")
    plt.plot(res.time, res.outputs, label="out")
    plt.plot(res.time, arr[1], label="out_real")
    plt.legend()
    plt.show()

def m14():
    tf,params = rand_stable_tf(TFType.ORD2_PER)
    clean_y,t_max = simulate_step(tf)
    # y = np.std(clean_y)
    y = add_noise(clean_y, NOISE_STD)
    y = smooth(y)
    # n_t,y = normalize_step_time(y, t_max)

    print(len(clean_y))
    print(len(y))

    plt.plot(y)
    plt.plot(clean_y)
    plt.show()

def m15():
    sys = System.get_random_sample(TFType.ORD1_ASTAT)
    sys.plot()
    sys = System.get_random_sample(TFType.ORD2_ASTAT)
    sys.plot()
    sys = System.get_random_sample(TFType.ORD2_ASTAT_T)
    sys.plot()

def m16():
    train_type(tftype=TFType.ORD2_APER, epochs=50, batch_size=64)

def m17():
    # tftype = TFType.ORD1        ; p1 = [1,1]        # ORD1
    # tftype = TFType.ORD1_ASTAT  ; p1 = [1]          # ORD1_ASTAT
    # tftype = TFType.ORD2_APER   ; p1 = [1,1,1]        # ORD2_APER
    # tftype = TFType.ORD2_PER   ; p1 = [1,1,0.5]    # ORD2_PER
    # tftype = TFType.ORD2_ASTAT  ; p1 = [1]          # ORD2_ASTAT
    tftype = TFType.ORD2_ASTAT_T; p1 = [1,1]        # ORD2_ASTAT_T
    
    t = torch.linspace(0,100,200)

    cusy = step_response_of_type(tftype, torch.tensor([p1]), t)

    tf = tf_from_params(tftype, p1)
    t = np.linspace(0,100,200)

    cty = ct.step_response(tf, t)

    plt.plot(t, cusy[0])
    plt.plot(t, cty.outputs)
    plt.show()

def m18():
    a = np.load(f"{TRAIN_SAVE_DIR}/ORD2_PER/X_step.npy")
    print(np.shape(a))
    print(np.max(a))
    print(np.min(a))

def m19():
    t = np.linspace(0,100,200)
    y3 = np.exp(-3 * t / t[-1])
    y2 = np.exp(-2 * t / t[-1])
    y1 = np.exp(-1 * t / t[-1])
    plt.plot(y3)
    plt.plot(y2)
    plt.plot(y1)
    plt.show()

def m20():
    s=System.get_random_sample(TFType.ORD2_APER, idx=0)
    sig = add_noise(signal=s.step_response, noise_std=0.2)
    plt.subplot(2,1,1)
    plt.plot(sig)

    sig = smooth(sig)
    plt.subplot(2,1,2)
    plt.plot(sig)

    plt.tight_layout()
    plt.show()

def m21():
    inp = np.concat([np.full(50,1),np.full(50,0),np.full(50,1),np.full(50,0)])
    t = np.linspace(0,50,len(inp))

    s = o1_tf_from_params(1,1)
    res = ct.forced_response(s,t,inp, initial_state=0)

    plt.plot(inp)
    plt.plot(res.outputs)
    plt.show()

def m22():
    # s1 = np.random.randint(0,200)
    # s2 = np.random.randint(0,200)
    # s3 = np.random.randint(0,200)
    # s4 = np.random.randint(0,200)
    s1 = int(np.random.randint(0,200)/4)
    s2 = int(np.random.randint(0,200)/4)
    s3 = int(np.random.randint(0,200)/4)
    s4 = int(np.random.randint(0,200)/4)

    inp = np.concat([np.full(s1,1),np.full(s2,0),np.full(s3,1),np.full(s4,0)])
    # t,inp = normalize_step_time(inp, 200)

    t = np.linspace(0,50,len(inp))

    s = o1_tf_from_params(1,1)
    res = ct.forced_response(s,t,inp, initial_state=0)

    plt.plot(inp)
    plt.plot(res.outputs)
    plt.show()

if __name__ == "__main__":
    save_n_rand(50, osc=False)
    save_n_rand(50, osc=True)