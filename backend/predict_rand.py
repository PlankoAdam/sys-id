import numpy as np
import matplotlib.pyplot as plt
import sys

from src.tf_predictor.predict import predict_with_model, load_model
from src.tftype_classifier.predict import predict_tftype
from src.core.utils import *
from src.core.types import TFType, System
from src.core.config import *

def pred_rand(tftype = None, idx = None, osc:bool = False):
    sys:System = System.get_random_sample(tftype, idx, osc=osc)

    print(f"Tend: {sys.time_end}")
    
    pred_tftype = predict_tftype(sys.step_response, osc=osc)
    print(f"True TF Type:\t\t{sys.tftype.name}\nPredicted TF Type:\t{pred_tftype.name}")

    model = load_model(pred_tftype, osc=osc)

    # ret = predict_and_plot_tf(pred_tftype, sys.step_response, true_params=sys.params, time_end=sys.time_end)
    pred = predict_with_model(model, pred_tftype, sys.step_response, sys.time_end, osc=osc)
    print(f"True TF Params:\t\t{sys.params}\nPredicted TF Params:\t{pred.params}")

    # err = np.abs(sys.step_response - pred.step_response)
    # err = np.sum(err)/np.max(sys.step_response)
    # # err = np.mean(err)
    # print(f"Plot error: {err}")

    t = np.linspace(0,sys.time_end,len(sys.step_response))
    return (t, sys.step_response, pred.step_response, pred.input, sys.tftype, pred_tftype, sys.params, pred.params)

def plot_single(tftype = None, idx = None, osc: bool = False):
    t, y, y_pred, inp, true_type, pred_type, true_params, pred_params = pred_rand(tftype, idx, osc)

    plt.xlabel('Time [s]')
    plt.ylabel('Output level')
    plt.grid()

    plt.plot(t, y, label='True response')
    plt.plot(t, y_pred, label='Predicted response')
    plt.plot(t, inp, label='Input')
    plt.legend()
    
    # Add text box with info
    info_text = f"True type:           {true_type.name}\n"
    info_text += f"Predicted type:      {pred_type.name}\n"
    info_text += f"True params:         {true_params}\n"
    info_text += f"Predicted params:    {pred_params}"
    
    plt.text(0.02, 1.1, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    plt.show()

def plot_n_rand(n: int):
    for i in range(n):
        t, y, y_pred, inp, true_type, pred_type, true_params, pred_params = pred_rand()

        plt.subplot(n, 1, i+1)
        plt.xlabel('Time [s]')
        plt.ylabel('Output level')
        plt.grid()

        plt.plot(t, y, label='True response')
        plt.plot(t, y_pred, label='Predicted response')
        plt.plot(t, inp, label='Input')
        plt.legend()
        
        # Add text box with info
        info_text = f"True type and params:           {true_type.name} {true_params}\n"
        info_text += f"Predicted type and params:      {pred_type.name} {pred_params}"
        
        plt.text(0.98, 0.82, info_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                family='monospace')
    # plt.tight_layout()
    plt.show()

def plot_noise_comp(tftype = None, idx = None):
    for i in range(3):
        t, y, y_pred, inp, true_type, pred_type, true_params, pred_params = pred_rand(tftype, 3*idx+i, osc=False)

        plt.subplot(3,1,i+1)
        plt.plot(t, y, label='True response')
        plt.plot(t, y_pred, label='Predicted response')
        plt.plot(t, inp, label='Input')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Output level')
        plt.grid()
        plt.legend()
        
        # Add text box with info
        info_text = f"True type and params:           {true_type.name} {true_params}\n"
        info_text += f"Predicted type and params:      {pred_type.name} {pred_params}"
        
        plt.text(0.98, 0.1, info_text, transform=plt.gca().transAxes,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                 family='monospace')
    plt.show()

def pred_real(fname):
    t, sr = load_and_process_csv(f"{DATA_DIR}/{fname}", threshold=0.01)

    pred_tftype = predict_tftype(sr)
    # pred_tftype = TFType.ORD2_APER
    print(f"Predicted TF Type: {pred_tftype.name}")

    model = load_model(pred_tftype)

    pred = predict_with_model(model, pred_tftype, sr, t[-1])
    print(f"Predicted TF Params: {pred.params}")

    plt.plot(sr)
    plt.plot(pred.step_response)
    plt.show()

def m1():
    try:
        tftype = TFType(int(sys.argv[1]))
        plot_single(tftype, int(sys.argv[2]))
    except IndexError:
        plot_single()
    
def m2():
    while True:
        plot_single()

def m3():
    pred_real(fname="data1.csv")
    pred_real(fname="data1i.csv")
    pred_real(fname="data2.csv")
    pred_real(fname="data3i.csv")
    pred_real(fname="data4i.csv")
    pred_real(fname="data5i.csv")

def m4():
    try:
        tftype = TFType(int(sys.argv[1]))
        plot_noise_comp(tftype, int(sys.argv[2]))
    except IndexError:
        plot_noise_comp()

if __name__ == "__main__":
    # plot_noise_comp(tftype=TFType.ORD2_PER, idx=5)
    plot_single(osc=True)