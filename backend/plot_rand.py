from src.core.types import TFType, System
from matplotlib import pyplot as plt

def comp():
    m=15
    t=TFType.ORD2_ASTAT
    s0:System = System.get_random_sample(t, idx=(3*m+0))
    s1:System = System.get_random_sample(t, idx=(3*m+1))
    s2:System = System.get_random_sample(t, idx=(3*m+2))

    plt.subplot(3,1,1)
    plt.plot(s0.step_response)
    plt.subplot(3,1,2)
    plt.plot(s1.step_response)
    plt.subplot(3,1,3)
    plt.plot(s2.step_response)
    plt.tight_layout()
    plt.show()

def comp_osc():
    m=222
    t=TFType.ORD2_PER
    s0:System = System.get_random_sample(t, idx=(3*m+0), osc=True)
    s1:System = System.get_random_sample(t, idx=(3*m+1), osc=True)
    s2:System = System.get_random_sample(t, idx=(3*m+2), osc=True)

    plt.subplot(3,1,1)
    plt.plot(s0.step_response)
    plt.subplot(3,1,2)
    plt.plot(s1.step_response)
    plt.subplot(3,1,3)
    plt.plot(s2.step_response)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # sys:System = System.get_random_sample(TFType.ORD2_APER, idx=2)
    # sys.plot()
    comp()
    # comp()