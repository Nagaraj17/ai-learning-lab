import math

def verify_scalar_step():
    print("--- Verifying Scalar Neuron Training Step ---")
    
    # 0. Initialization
    x = 2.0
    y = 10.0
    W = 3.0
    lr = 0.1
    print(f"Inputs: x={x}, y={y}, W={W}, lr={lr}")
    
    # 1. Forward Pass
    prediction = x * W
    print(f"Prediction: {prediction} (Expected: 6.0)")
    assert math.isclose(prediction, 6.0), "Prediction mismatch!"
    
    # 2. Loss Calculation
    loss = 0.5 * (prediction - y)**2
    print(f"Loss: {loss} (Expected: 8.0)")
    assert math.isclose(loss, 8.0), "Loss mismatch!"
    
    # 3. Backpropagation (Gradient)
    gradient = (prediction - y) * x
    print(f"Gradient: {gradient} (Expected: -8.0)")
    assert math.isclose(gradient, -8.0), "Gradient mismatch!"
    
    # 4. Parameter Update
    W_new = W - (lr * gradient)
    print(f"New Weight: {W_new} (Expected: 3.8)")
    assert math.isclose(W_new, 3.8), "New Weight mismatch!"
    
    # 5. Verification (New Prediction and Loss)
    new_prediction = x * W_new
    new_loss = 0.5 * (new_prediction - y)**2
    print(f"New Prediction: {new_prediction} (Expected: 7.6)")
    assert math.isclose(new_prediction, 7.6), "New prediction mismatch!"
    print(f"New Loss: {new_loss:.2f} (Expected: 2.88)")
    assert math.isclose(new_loss, 2.88), "New loss mismatch!"
    
    print("SUCCESS: All scalar calculations match the markdown exercise perfectly.")

if __name__ == "__main__":
    verify_scalar_step()
