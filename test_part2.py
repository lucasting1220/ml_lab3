"""
Tests for Lab 3, Part 2. Don't edit this file.

    python test_part2.py               run every test on your lab3_part2.py
    python test_part2.py softmax       run only the tests whose name contains "softmax"

Each test prints ✅ / ❌ (with what you got vs. what was expected), or ⏭ if you haven't written it yet.
"""
import sys, importlib
import numpy as np
from lab_utils import enc, EOS, N_HEADS, params, check, show_top_k, run_tests, check_against_gpt2

# Load your functions from lab3_part2.py  (instructors: add --solution to test the answer key)
_module = importlib.import_module("lab3_part2_solution" if "--solution" in sys.argv else "lab3_part2")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
_filters = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]

# ============================================================================
# Tests
# ============================================================================

def test_mask():
    check("causal_mask(3)", causal_mask(3), [[0, 1, 1], [0, 0, 1], [0, 0, 0]])


def test_attention():
    # Lecture 2's worked example: The, old, cat, sat (after positional embeddings), one token per ROW
    E = np.array([[1.90, 0.10, 1.20, 0.40],    # The
                  [1.14, 1.34, 1.10, 0.61],    # old
                  [1.61, 1.51, 1.90, 0.22],    # cat
                  [0.24, 1.19, 1.80, 0.73]])   # sat
    W_Q = np.array([[0.1, 0.2, 0.3, 0.1], [0.4, 0.1, 0.2, 0.3], [0.2, 0.3, 0.1, 0.4], [0.1, 0.4, 0.3, 0.2]])
    W_K = np.array([[0.2, 0.1, 0.4, 0.2], [0.1, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1, 0.3], [0.1, 0.4, 0.3, 0.2]])
    V   = np.array([[0.5, 0.2, 0.6, 0.3], [0.4, 0.7, 0.3, 0.5], [0.8, 0.6, 0.7, 0.4], [0.3, 0.5, 0.8, 0.6]])
    Q = E @ W_Q.T    # lecture computed W_Q · e for each column e; with rows that's E @ W_Q.T
    K = E @ W_K.T

    out, weights = attention(Q, K, V, mask=True)
    check("attention weights for 'sat'", weights[3], [0.157, 0.233, 0.372, 0.238], atol=1e-3)
    check("ΔE for 'sat'", out[3], [0.541, 0.537, 0.615, 0.455], atol=1e-3)
    check("new embedding for 'sat' (E + ΔE)", E[3] + out[3], [0.781, 1.727, 2.415, 1.185], atol=1e-3)
    check("no token attends to the future", np.triu(weights, k=1), np.zeros((4, 4)))
    check("each query's weights sum to 1", weights.sum(-1), np.ones(4))

    # Works with a heads dimension too?
    Qh, Kh, Vh = np.random.default_rng(0).normal(size=(3, 2, 5, 4))   # 2 heads, 5 tokens, d_k = 4
    check("attention works on (heads, n, d_k)", attention(Qh, Kh, Vh)[0][1], attention(Qh[1], Kh[1], Vh[1])[0])


def test_split_merge():
    x = np.arange(12).reshape(2, 6)    # 2 tokens, d = 6
    check("split_heads shape", split_heads(x, 3).shape, (3, 2, 2))
    check("head 1 gets columns 2-3", split_heads(x, 3)[1], [[2, 3], [8, 9]])
    check("merge_heads undoes split_heads", merge_heads(split_heads(x, 3)), x)


def test_mha():
    rng = np.random.default_rng(2)
    d = 8
    toy_block = {"W_qkv": rng.normal(size=(d, 3 * d)), "b_qkv": rng.normal(size=3 * d),
                 "W_O": rng.normal(size=(d, d)), "b_O": rng.normal(size=d)}
    x = rng.normal(size=(3, d))
    out, weights = multi_head_attention(x, toy_block, n_heads=2)
    check("multi_head_attention output on a toy block", out,
          [[ 4.245598,  8.788397,  7.672221, -7.505903,  4.306118,  8.589963, -0.766082,  0.718587],
     [ 4.267835,  8.776392,  7.654946, -7.517776,  4.302150,  8.581655, -0.772932,  0.704389],
     [ 8.054230,  0.666530, -0.765144, -4.981527, -1.583426,  2.245614, -4.031463, -9.511483]])
    check("weights shape is (n_heads, n, n)", weights.shape, (2, 3, 3))


def test_plugin():
    check_against_gpt2(forward)    # full GPT-2 this time, attention included


def test_head_scores():
    w = np.zeros((2, 4, 4))          # one layer: 2 heads, 4 tokens
    w[0, 0, 0] = 1                   # head 0: every token looks at the token right before it
    for i in range(1, 4):
        w[0, i, i - 1] = 1
    w[1] = np.eye(4)                 # head 1: every token looks only at itself
    check("head_scores shape is (n_layers, n_heads)", head_scores([w, w, w], offset=1).shape, (3, 2))
    check("offset 1: previous-token head scores 1, self-attending head scores 0", head_scores([w], offset=1), [[1.0, 0.0]])
    check("offset 2: neither head looks back 2 tokens", head_scores([w], offset=2), [[0.0, 0.0]])
    check("offset 0: self-attending head scores 1", head_scores([w], offset=0), [[0.25, 1.0]])


TESTS = [
    ('Causal mask', test_mask),
    ('Attention (Lecture 2 example)', test_attention),
    ('Split / merge heads', test_split_merge),
    ('Multi-head attention', test_mha),
    ('Plugged into GPT-2', test_plugin),
    ('Head scores', test_head_scores),
]

# ============================================================================
# Experiments (run after all tests, when no filter is given)
# ============================================================================

def experiments():
    for prompt in ["The old cat sat on the", "The old dog sat on the"]:
        probs = softmax(forward(enc.encode(prompt), params)[-1])
        print(prompt)
        show_top_k(probs)

    for T in [0.3, 1.0, 1.5]:
        print(f"\nT = {T}:", generate("The old cat sat on the", max_new_tokens=20, T=T))


def find_heads():
    _, cache = forward_with_cache(enc.encode("The old cat sat on the mat because it was tired and wanted to sleep."))
    print("Heads that look at the PREVIOUS token:")
    top_heads(head_scores(cache, offset=1))

    # Induction: a random sequence of tokens, repeated twice
    rng = np.random.default_rng(0)
    L = 25
    V_size = params["W_E"].shape[0]
    seq = [int(t) for t in rng.integers(V_size // 50, V_size // 5, size=L)]
    ids = seq + seq
    logits, cache = forward_with_cache(ids)
    probs = softmax(logits)

    def avg_loss(positions):   # cross-entropy of predicting ids[i + 1] from position i
        return -np.mean([np.log(probs[i, ids[i + 1]]) for i in positions])

    print(f"\nLoss on 1st copy: {avg_loss(range(0, L - 1)):.2f}")
    print(f"Loss on 2nd copy: {avg_loss(range(L, 2 * L - 1)):.2f}")
    print("Heads that look at the token right AFTER the earlier copy (induction heads):")
    top_heads(head_scores(cache, offset=L - 1))


if __name__ == "__main__":
    selected = [(n, f) for n, f in TESTS if not _filters or any(x in n.lower() for x in _filters)]
    if not selected:
        print("No test names match", _filters, "- options:", [n for n, _ in TESTS])
    run_tests(selected)
    if not _filters:
        print("\n=== Experiments: compare with Part 1 ===")
        run_tests([("Experiments", experiments)])
        print("\n=== Section 4: looking inside the heads ===")
        run_tests([("Head search", find_heads)])
