HIDDEN1 = 256
HIDDEN2 = 128
LEARNING_RATE = 3e-3


def init_mlp_params(key, input_dim=INPUT_DIM, hidden1=HIDDEN1, hidden2=HIDDEN2, num_classes=NUM_CLASSES):
    """Initialize a 3-layer MLP with He-style weight scaling and zero biases."""
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (input_dim, hidden1), dtype=jnp.float32) * math.sqrt(2.0 / input_dim),
        "b1": jnp.zeros((hidden1,), dtype=jnp.float32),
        "w2": jax.random.normal(k2, (hidden1, hidden2), dtype=jnp.float32) * math.sqrt(2.0 / hidden1),
        "b2": jnp.zeros((hidden2,), dtype=jnp.float32),
        "w3": jax.random.normal(k3, (hidden2, num_classes), dtype=jnp.float32) * math.sqrt(2.0 / hidden2),
        "b3": jnp.zeros((num_classes,), dtype=jnp.float32),
    }


def mlp(params, x, compute_dtype=jnp.float32):
    """Forward pass: cast inputs/params to `compute_dtype`, two GELU hidden layers, then cast logits back to float32."""
    x = x.astype(compute_dtype)
    w1 = params["w1"].astype(compute_dtype)
    b1 = params["b1"].astype(compute_dtype)
    w2 = params["w2"].astype(compute_dtype)
    b2 = params["b2"].astype(compute_dtype)
    w3 = params["w3"].astype(compute_dtype)
    b3 = params["b3"].astype(compute_dtype)

    x = jax.nn.gelu(x @ w1 + b1)
    x = jax.nn.gelu(x @ w2 + b2)
    logits = x @ w3 + b3
    return logits.astype(jnp.float32)


def cross_entropy_loss(params, batch, compute_dtype=jnp.float32):
    """Scalar softmax cross-entropy loss."""
    x, y = batch
    logits = mlp(params, x, compute_dtype=compute_dtype)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def loss_with_metrics(params, batch, compute_dtype=jnp.float32):
    """Same loss, but also returns batch accuracy in an aux dict."""
    x, y = batch
    logits = mlp(params, x, compute_dtype=compute_dtype)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, {"accuracy": accuracy}


params = init_mlp_params(jax.random.key(1))
params = jax.device_put(params, device)

rows = []
for name, value in params.items():
    rows.append((name, value.shape, value.dtype, value.devices()))
show_table(["Parameter", "Shape", "Dtype", "Devices"], rows, title=f"MLP parameters: {count_params(params):,} trainable values")
