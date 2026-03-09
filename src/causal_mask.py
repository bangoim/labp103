import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i][j] = -np.inf
    return mask


if __name__ == "__main__":
    seq_len = 5
    d_k = 8

    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    scores = Q @ K.T / np.sqrt(d_k)

    M = create_causal_mask(seq_len)
    masked_scores = scores + M

    attention_weights = softmax(masked_scores)

    print("Máscara Causal M:")
    print(M)
    print("\nScores (QK^T / sqrt(d_k)) + M:")
    print(masked_scores)
    print("\nPesos de Atenção após Softmax:")
    print(attention_weights)
    print("\nVerificação: posições futuras são 0.0?")
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            print(f"  posição [{i},{j}] = {attention_weights[i][j]:.6f}")
