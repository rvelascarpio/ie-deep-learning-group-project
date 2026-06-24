import tensorflow as tf
import numpy as np
import scipy.ndimage

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D) or 'conv1d' in layer.name.lower():
            return layer.name
    raise ValueError("Could not find a 1D convolutional layer in the model.")

def compute_gradcam_1d(model, sig_batch, target_class_idx):
    """
    Computes a 1D Grad-CAM heatmap for a given model and input signal.
    """
    last_conv_layer_name = get_last_conv_layer_name(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(sig_batch)
        loss = predictions[:, target_class_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling over the time dimension (axis 1)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    # Weight the convolution outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # ReLU to keep only positive influences
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize the heatmap
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val
        
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match original signal length (1000)
    original_len = sig_batch.shape[1]
    zoom_factor = original_len / len(heatmap)
    heatmap = scipy.ndimage.zoom(heatmap, zoom_factor, order=1) # Linear interpolation
    
    # Ensure values are between 0 and 1 after interpolation
    heatmap = np.clip(heatmap, 0, 1)
    
    return heatmap
