"""Feed-forward ReLU network container.

Convention used throughout the codebase
---------------------------------------
    z^(0)   = x
    zhat^(i)= W[i-1] @ z^(i-1) + b[i-1]      for i = 1 .. L
    z^(i)   = relu(zhat^(i))                 for i = 1 .. L-1
    output  = zhat^(L)

So `W[i-1]` produces pre-activation layer `i`.  Layer indices for
pre-activations are 1-based; index 0 is never a pre-activation.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, weights, biases, name="mlp"):
        assert len(weights) == len(biases)
        self.W = [np.asarray(w, dtype=np.float64) for w in weights]
        self.b = [np.asarray(v, dtype=np.float64) for v in biases]
        self.name = name
        self.L = len(self.W)                      # index of output pre-activation
        self.widths = [self.W[0].shape[1]] + [w.shape[0] for w in self.W]
        # cached absolute weights for the implied-stabilisation feature (F2)
        self.absW = [np.abs(w) for w in self.W]

    @property
    def n_in(self):
        return self.widths[0]

    @property
    def n_out(self):
        return self.widths[-1]

    def hidden_sizes(self):
        return self.widths[1:-1]

    def forward(self, x):
        """x: (batch, n_in) -> logits (batch, n_out)"""
        h = np.atleast_2d(x)
        for i in range(self.L):
            h = h @ self.W[i].T + self.b[i]
            if i < self.L - 1:
                h = np.maximum(h, 0.0)
        return h

    def forward_with_grad(self, x, c):
        """Value and gradient of  c . f(x)  for each row of x.

        x: (batch, n_in), c: (n_out,) -> (values (batch,), grads (batch, n_in))
        """
        h = np.atleast_2d(x)
        acts = []
        for i in range(self.L):
            pre = h @ self.W[i].T + self.b[i]
            if i < self.L - 1:
                acts.append(pre > 0)
                h = np.maximum(pre, 0.0)
            else:
                h = pre
        val = h @ c
        g = np.broadcast_to(c, (x.shape[0], c.shape[0])).copy()
        for i in range(self.L - 1, -1, -1):
            g = g @ self.W[i]
            if i > 0:
                g = g * acts[i - 1]
        return val, g


def train_mlp(rng, n_in, hidden, n_out, X, Y, epochs=250, lr=0.06, name="mlp"):
    """Tiny numpy SGD trainer.  Enough to get non-trivial decision boundaries."""
    sizes = [n_in] + list(hidden) + [n_out]
    W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i + 1], sizes[i]))
         for i in range(len(sizes) - 1)]
    b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    n = X.shape[0]
    Yoh = np.eye(n_out)[Y]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, 64):
            xb, yb = X[idx[s:s + 64]], Yoh[idx[s:s + 64]]
            hs, pres = [xb], []
            h = xb
            for i in range(len(W)):
                pre = h @ W[i].T + b[i]
                pres.append(pre)
                h = np.maximum(pre, 0.0) if i < len(W) - 1 else pre
                hs.append(h)
            p = np.exp(h - h.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            d = (p - yb) / xb.shape[0]
            for i in range(len(W) - 1, -1, -1):
                gW = d.T @ hs[i]
                gb = d.sum(0)
                if i > 0:
                    d = (d @ W[i]) * (pres[i - 1] > 0)
                W[i] -= lr * gW
                b[i] -= lr * gb
    return MLP(W, b, name=name)
