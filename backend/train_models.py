from src.tf_predictor.train_model import train_all
from src.tftype_classifier.train_model import train as train_tftype_classifier

if __name__=="__main__":
    print("Training STEP models...")
    train_tftype_classifier(epochs=24, batch_size=256)
    train_all(epochs=60, batch_size=128)
    
    print("Training OSC models...")
    train_tftype_classifier(epochs=24, batch_size=256, osc=True)
    train_all(epochs=60, batch_size=128, osc=True)