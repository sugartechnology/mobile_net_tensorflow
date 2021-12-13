
import tensorflow as tf
from mobile_unet import load_model, weights_path



if __name__=="__main__":  model = load_model(weights_path=weights_path)

tf.saved_model.save(model, 'model_unet/save')