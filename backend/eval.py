from src.tftype_classifier.eval_model import eval_model as eval_tftype_classifier_model
from src.tf_predictor.eval_model import eval_all as eval_all_tf_predictor_models

if __name__ == "__main__":
    osc = True
    eval_tftype_classifier_model(osc=osc)
    eval_all_tf_predictor_models(osc=osc)