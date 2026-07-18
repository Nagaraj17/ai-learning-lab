import numpy as np

def verify_forward_pass():
    print("--- Verifying Manual Forward Pass ---")
    
    # Initialization
    x = np.array([1.0, 0.0])
    target = 1
    
    W1 = np.array([
        [ 0.5, -0.2],
        [-0.1,  0.3]
    ])
    b1 = np.array([0.0, 0.0])
    
    W2 = np.array([
        [ 0.4,  0.1],
        [-0.3,  0.6]
    ])
    b2 = np.array([0.0, 0.0])
    
    # 1. Linear Transformation 1
    z1 = x @ W1 + b1
    print(f"z1: {z1} (Shape: {z1.shape})")
    assert np.allclose(z1, [0.5, -0.2], atol=1e-4)
    
    # 2. Activation 1 (tanh)
    h1 = np.tanh(z1)
    print(f"h1: {h1} (Shape: {h1.shape})")
    assert np.allclose(h1, [0.4621, -0.1974], atol=1e-4)
    
    # 3. Linear Transformation 2 (Logits)
    logits = h1 @ W2 + b2
    print(f"logits: {logits} (Shape: {logits.shape})")
    assert np.allclose(logits, [0.2440, -0.0722], atol=1e-4)
    
    # 4. Output Activation (Softmax)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    print(f"probs: {probs} (Shape: {probs.shape})")
    assert np.allclose(probs, [0.5784, 0.4216], atol=1e-4)
    
    # 5. Cross-Entropy Loss
    loss = -np.log(probs[target])
    print(f"loss: {loss:.4f} (Shape: {loss.shape})")
    assert np.allclose(loss, 0.8637, atol=1e-4)
    
    print("SUCCESS: All forward pass calculations match the markdown exercise perfectly.")

if __name__ == "__main__":
    verify_forward_pass()
