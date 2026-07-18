import numpy as np

def verify_backpropagation():
    print("--- Verifying Manual Backpropagation ---")
    
    # Stored Values
    target = 1
    x = np.array([[1.0, 0.0]]) # Note: Row vector for correct outer product shapes
    h1 = np.array([[0.462117, -0.197375]]) 
    probs = np.array([[0.57835, 0.42165]])
    
    W2 = np.array([
        [ 0.4,  0.1],
        [-0.3,  0.6]
    ])
    
    # 1. Gradient at Output
    true_one_hot = np.array([[0.0, 1.0]])
    d_logits = probs - true_one_hot
    print(f"d_logits: {d_logits} (Shape: {d_logits.shape})")
    assert np.allclose(d_logits, [[0.5784, -0.5784]], atol=1e-4)
    
    # 2. Gradients for Layer 2
    dW2 = h1.T @ d_logits
    print(f"dW2:\n{dW2} (Shape: {dW2.shape})")
    assert np.allclose(dW2, [[0.2673, -0.2673], [-0.1142, 0.1142]], atol=1e-4)
    
    db2 = np.sum(d_logits, axis=0) # Sum across batch
    print(f"db2: {db2} (Shape: {db2.shape})")
    assert np.allclose(db2, [0.5784, -0.5784], atol=1e-4)
    
    # 3. Gradient at Hidden Layer
    d_hidden = d_logits @ W2.T
    print(f"d_hidden: {d_hidden} (Shape: {d_hidden.shape})")
    assert np.allclose(d_hidden, [[0.1736, -0.5205]], atol=1e-4)
    
    # 4. Gradient through Activation (tanh)
    d_z1 = d_hidden * (1 - h1**2)
    print(f"d_z1: {d_z1} (Shape: {d_z1.shape})")
    assert np.allclose(d_z1, [[0.1365, -0.5002]], atol=1e-4)
    
    # 5. Gradients for Layer 1
    dW1 = x.T @ d_z1
    print(f"dW1:\n{dW1} (Shape: {dW1.shape})")
    assert np.allclose(dW1, [[0.1365, -0.5002], [0.0000, 0.0000]], atol=1e-4)
    
    db1 = np.sum(d_z1, axis=0)
    print(f"db1: {db1} (Shape: {db1.shape})")
    assert np.allclose(db1, [0.1365, -0.5002], atol=1e-4)
    
    print("SUCCESS: All backward pass calculations match the markdown exercise perfectly.")

if __name__ == "__main__":
    verify_backpropagation()
