import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_attention(encoder_out, decoder_state):
    """
    Calcula o Scaled Dot-Product Attention cruzado entre decoder e encoder.

    Args:
        encoder_out: saída do encoder, shape [batch_size, seq_len_encoder, d_model]
        decoder_state: estado atual do decoder, shape [batch_size, seq_len_decoder, d_model]

    Returns:
        output: resultado da atenção, shape [batch_size, seq_len_decoder, d_model]
        weights: pesos de atenção, shape [batch_size, seq_len_decoder, seq_len_encoder]
    """
    d_model = encoder_out.shape[-1]

    np.random.seed(0)
    W_Q = np.random.randn(d_model, d_model) * 0.01
    W_K = np.random.randn(d_model, d_model) * 0.01
    W_V = np.random.randn(d_model, d_model) * 0.01

    Q = decoder_state @ W_Q
    K = encoder_out @ W_K
    V = encoder_out @ W_V

    d_k = d_model
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    weights = softmax(scores)

    output = weights @ V
    return output, weights


if __name__ == "__main__":
    batch_size = 1
    seq_len_frances = 10
    seq_len_ingles = 4
    d_model = 512

    np.random.seed(42)
    encoder_output = np.random.randn(batch_size, seq_len_frances, d_model)
    decoder_state = np.random.randn(batch_size, seq_len_ingles, d_model)

    output, weights = cross_attention(encoder_output, decoder_state)

    print(f"encoder_output shape: {encoder_output.shape}")
    print(f"decoder_state shape:  {decoder_state.shape}")
    print(f"output shape:         {output.shape}")
    print(f"weights shape:        {weights.shape}")
    print(f"\nPesos de atenção (cada linha do decoder sobre o encoder):")
    for i in range(seq_len_ingles):
        print(f"  decoder token {i} -> encoder: {weights[0, i, :]}")
        print(f"    soma = {weights[0, i, :].sum():.6f}")
