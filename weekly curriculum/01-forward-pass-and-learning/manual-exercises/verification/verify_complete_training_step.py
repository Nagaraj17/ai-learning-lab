import numpy as np

def verify_complete_step():
    print("--- Verifying Complete Training Step ---")
    
    lr = 0.1
    
    W2 = np.array([
        [ 0.4,  0.1],
        [-0.3,  0.6]
    ])
    dW2 = np.array([
        [ 0.2673, -0.2673], 
        [-0.1142,  0.1142]
    ])
    
    W1 = np.array([
        [ 0.5, -0.2],
        [-0.1,  0.3]
    ])
    dW1 = np.array([
        [ 0.1365, -0.5002], 
        [ 0.0000,  0.0000]
    ])
    
    # 1. Update W2
    W2_new = W2 - (lr * dW2)
    print(f"W2_new:\n{W2_new}")
    assert np.allclose(W2_new, [[0.3733, 0.1267], [-0.2886, 0.5886]], atol=1e-4)
    
    # 2. Update W1
    W1_new = W1 - (lr * dW1)
    print(f"W1_new:\n{W1_new}")
    assert np.allclose(W1_new, [[0.4863, -0.1500], [-0.1000, 0.3000]], atol=1e-4)
    
    print("SUCCESS: All parameter updates match the markdown exercise perfectly.")

if __name__ == "__main__":
    verify_complete_step()
