import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def generate_next_token(current_sequence, encoder_out, vocab, d_model=512):
    np.random.seed(len(current_sequence))

    seq_len = len(current_sequence)
    decoder_state = np.random.randn(seq_len, d_model)

    W_Q = np.random.randn(d_model, d_model) * 0.01
    W_K = np.random.randn(d_model, d_model) * 0.01
    W_V = np.random.randn(d_model, d_model) * 0.01

    Q = decoder_state @ W_Q
    K = encoder_out.reshape(-1, d_model) @ W_K
    V = encoder_out.reshape(-1, d_model) @ W_V

    scores = Q @ K.T / np.sqrt(d_model)
    weights = softmax(scores)
    context = weights @ V

    last_hidden = context[-1]

    W_proj = np.random.randn(d_model, len(vocab)) * 0.01
    logits = last_hidden @ W_proj

    probs = softmax(logits)
    return probs


if __name__ == "__main__":
    vocab = [f"word_{i}" for i in range(10000)]
    vocab[0] = "<START>"
    vocab[1] = "<EOS>"
    vocab[2] = "O"
    vocab[3] = "rato"
    vocab[4] = "roeu"
    vocab[5] = "a"
    vocab[6] = "roupa"
    vocab[7] = "do"
    vocab[8] = "rei"

    d_model = 512
    seq_len_encoder = 10
    np.random.seed(99)
    encoder_out = np.random.randn(1, seq_len_encoder, d_model)

    current_sequence = ["<START>"]
    max_steps = 20

    print("Iniciando loop de inferência auto-regressivo...\n")

    step = 0
    while step < max_steps:
        probs = generate_next_token(current_sequence, encoder_out, vocab)

        next_token_idx = np.argmax(probs)
        next_token = vocab[next_token_idx]

        current_sequence.append(next_token)

        print(f"Passo {step + 1}: token gerado = '{next_token}' (idx={next_token_idx}, prob={probs[next_token_idx]:.6f})")

        if next_token == "<EOS>":
            break

        step += 1

    print(f"\nFrase final gerada: {' '.join(current_sequence)}")
