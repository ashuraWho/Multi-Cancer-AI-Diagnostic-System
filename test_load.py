import tensorflow as tf
import time
import sys
import os

def test_load():
    model_path = "multi_cancer_ai/models/model_optimized.tflite"
    print(f"Testing load of: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file missing!")
        return

    try:
        # Initialize Interpreter
        # Standard TF initialization
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("SUCCESS: Model loaded and tensors allocated!")
        
        # Test Inference with dummy data
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input: {input_details[0]['shape']}")
        
        import numpy as np
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print("SUCCESS: Inference ran!")
        print(f"Output: {output_data}")

    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
