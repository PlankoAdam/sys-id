from src.tftype_classifier.predict import load_model as load_classifier_model, predict_tftype_with_model
from src.tf_predictor.predict import load_model as load_regression_model, predict_with_model
from src.core.types import *
from src.core.utils import *

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import control as ct

classifier_model = load_classifier_model()
classifier_model_osc = load_classifier_model(osc=True)
regression_models_dict = {tftype: load_regression_model(tftype) for tftype in TFType}
regression_models_dict_osc = {tftype: load_regression_model(tftype, osc=True) for tftype in TFType}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictReq(BaseModel):
    step_response: list[float]
    Tmax: float
    tftype: str | None = None
    strip_zeros: bool = True
    strip_threshold: float = 0.001
    step_mag: float = 1.0
    input_func: list[float] | None = None
    osc_inp: bool = False

class TFParams(BaseModel):
    K: float
    T1: float | None = None
    T2: float | None = None
    b: float | None = None

class PredictRes(BaseModel):
    pred_tftype: str
    tf_params: TFParams
    step_mag: float
    pred_step_response: list[float]
    norm_step_response: list[float]
    norm_t: list[float]
    norm_in: list[float] | None = None

def tfparams_from_arr(tftype: TFType, params: list[float]) -> TFParams:
    if tftype == TFType.ORD1:
        return TFParams(K=params[0], T1=params[1])
    elif tftype == TFType.ORD1_ASTAT:
        return TFParams(K=params[0])
    elif tftype == TFType.ORD2_APER:
        return TFParams(K=params[0], T1=params[1], T2=params[2])
    elif tftype == TFType.ORD2_PER:
        return TFParams(K=params[0], T1=params[1], b=params[2])
    elif tftype == TFType.ORD2_ASTAT:
        return TFParams(K=params[0])
    elif tftype == TFType.ORD2_ASTAT_T:
        return TFParams(K=params[0], T1=params[1])
    return None

@app.post("/predict")
def predict(data: PredictReq) -> PredictRes:
    t = np.linspace(0.0, data.Tmax, len(data.step_response))
    sr = data.step_response
    inpfunc = data.input_func

    if data.strip_zeros:
        stripped_t, stripped_sr, inpfunc = strip_leading_zeros(t, sr, input_func=inpfunc, threshold=data.strip_threshold)
    norm_t, norm_sr = normalize_step_time(stripped_sr, stripped_t[-1])

    
    if data.tftype:
        try:
            tftype: TFType = TFType[data.tftype.upper()]
        except Exception as e:
            return JSONResponse(content={"error":f"Invalid tftype string, valid values are: {TFType._member_names_}", "exception":f"{e}"}, status_code=400)
    else:
        class_model_to_use = classifier_model_osc if data.osc_inp else classifier_model
        tftype = predict_tftype_with_model(norm_sr, model=class_model_to_use)

    shifted_t = stripped_t-stripped_t[0]
        
    reg_model_to_use = regression_models_dict_osc.get(tftype) if data.osc_inp else regression_models_dict.get(tftype)
    _, params_pred, tf_pred, _ = predict_with_model(reg_model_to_use, tftype, norm_sr, shifted_t[-1], osc=data.osc_inp)

    step_mag = data.step_mag if data.step_mag and data.step_mag > 0 else (inpfunc[-1] if inpfunc else 1)
    # simulate response with given input function
    sz = len(stripped_t)
    szq = int(sz/4)
    in_func = (inpfunc or np.full(sz, 1)*step_mag) if not data.osc_inp else np.concat([np.full(szq, 1),np.full(szq, 0),np.full(szq, 1),np.full(szq, 0)])*step_mag
    in_func = np.array(in_func, dtype=float)
    in_func = np.concat([in_func, np.full(sz-len(in_func), in_func[-1])])
    # print(f"in_func: {len(in_func)}")
    # print(f"stripped_t : {len(stripped_t)}")
    # norm_t, in_func = normalize_step_time(inpfunc, norm_t[-1])
    res = ct.forced_response(tf_pred, shifted_t, in_func / step_mag)
    sr_pred = res.outputs
    
    params_pred[0] = params_pred[0] / step_mag
    
    return PredictRes(
        pred_tftype=tftype.name,
        tf_params=tfparams_from_arr(tftype=tftype, params=params_pred),
        step_mag=step_mag,
        pred_step_response=sr_pred,
        norm_step_response=stripped_sr,
        norm_t=stripped_t,
        norm_in=in_func
        )
    
class StepReq(BaseModel):
    tftype: str
    params: TFParams
    Tmax: float
    Tmin: float = 0.0
    step_mag: float = 1.0
    input_func: list[float] | None = None
    
class StepRes(BaseModel):
    step_response: list[float]
    
@app.post("/step")
def sr_from_params(data: StepReq) -> StepRes:
    try:
        tftype: TFType = TFType[data.tftype.upper()]
    except Exception as e:
        return JSONResponse(content={"error":f"Invalid tftype string, valid values are: {TFType._member_names_}", "exception":f"{e}"}, status_code=400)
    
    try:
        if tftype == TFType.ORD1:
            tf = o1_tf_from_params(K=data.params.K, T=data.params.T1)
        elif tftype == TFType.ORD1_ASTAT:
            tf = o1astat_tf_from_params(K=data.params.K)
        elif tftype == TFType.ORD2_APER:
            tf = o2aper_tf_from_params(K=data.params.K, T1=data.params.T1, T2=data.params.T2)
        elif tftype == TFType.ORD2_PER:
            tf = o2per_tf_from_params(K=data.params.K, T=data.params.T1, b=data.params.b)
        elif tftype == TFType.ORD2_ASTAT:
            tf = o2astat_tf_from_params(K=data.params.K)
        elif tftype == TFType.ORD2_ASTAT_T:
            tf = o2astat_t_tf_from_params(K=data.params.K, T=data.params.T1)
    except Exception as e:
        return JSONResponse(content={"error":"Wrong parameters for given tftype"})
    
    t = np.linspace(0, data.Tmax, TIME_POINTS)

    # simulate response with given input function
    if data.input_func:
        in_func = data.input_func

        first_idx = find_first_non_zero_idx(in_func)
        # print(f"first idx: {first_idx}")
        # t = t[first_idx:]
        in_func = in_func[first_idx:]

        t, in_func = normalize_step_time(in_func, data.Tmax, norm_start_time=data.Tmin, time_points=len(data.input_func))

        res = ct.forced_response(tf, t, in_func / data.step_mag)
        sr = res.outputs
    else:
        _,sr = ct.step_response(tf, T=t)
    
    return StepRes(step_response=sr)
    
@app.middleware("http")
async def handle_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(content={"error":f"{e}"}, status_code=500)