@jax.jit
def sgd_step_jit(params, batch):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, batch)
    new_params = jax.tree.map(lambda p, g: p - LEARNING_RATE * g, params, grads)
    return new_params, loss


def batch_at(step):
    """Pick batch `step % num_batches`."""
    i = step % x_train_batches.shape[0]
    return x_train_batches[i], y_train_batches[i]


def time_loop(step_fn, params, steps):
    """Run `step_fn` for `steps` iterations and time it."""
    start = time.perf_counter()
    loss = None
    for step in range(steps):
        params, loss = step_fn(params, batch_at(step))
    params, loss = block_tree((params, loss))
    elapsed = time.perf_counter() - start
    return params, loss, elapsed


params_warm, loss_warm = sgd_step_jit(params, first_batch)
block_tree((params_warm, loss_warm))

EAGER_STEPS = 20
JIT_STEPS = 100

# sgd_step returns (params, loss, grads).
_, eager_loss, eager_elapsed = time_loop(lambda p, b: sgd_step(p, b)[:2], params, EAGER_STEPS)
_, jit_loss, jit_elapsed = time_loop(sgd_step_jit, params, JIT_STEPS)

eager_rate = EAGER_STEPS * BATCH_SIZE / eager_elapsed
jit_rate = JIT_STEPS * BATCH_SIZE / jit_elapsed

show_table(
    ["Mode", "Steps", "Final loss", "Elapsed seconds", "Examples/sec"],
    [
        ("Python dispatch", EAGER_STEPS, f"{float(eager_loss):.4f}", f"{eager_elapsed:.3f}", f"{eager_rate:,.0f}"),
        ("jitted step", JIT_STEPS, f"{float(jit_loss):.4f}", f"{jit_elapsed:.3f}", f"{jit_rate:,.0f}"),
    ],
    title="Cached training-step throughput",
    aligns=["left", "right", "right", "right", "right"],
)
show_bars([("Python dispatch", eager_rate), ("jitted step", jit_rate)], "Examples per second", "examples/s")
