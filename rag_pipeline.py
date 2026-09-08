def sgd_step(params, batch):
    """One un-jitted SGD update and returns (new_params, loss, grads)."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, batch)
    new_params = jax.tree.map(lambda p, g: p - LEARNING_RATE * g, params, grads)
    return new_params, loss, grads


grads_only = jax.grad(cross_entropy_loss)(params, first_batch)
loss_value, grads = jax.value_and_grad(cross_entropy_loss)(params, first_batch)
loss_value, grads, grads_only = block_tree((loss_value, grads, grads_only))

rows = []
for name in params:
    rows.append((name, params[name].shape, grads[name].shape, grads[name].dtype))
show_table(["Leaf", "Param shape", "Grad shape", "Grad dtype"], rows, title="Gradient tree matches the parameter tree")

grad_difference = tree_l2_norm(jax.tree.map(lambda a, b: a - b, grads, grads_only))

print(f"loss before update: {float(loss_value):.4f}")
print(f"gradient L2 norm:   {float(tree_l2_norm(grads)):.4f}")
print(f"grad vs value_and_grad difference: {float(grad_difference):.6f}")

params_after_one, loss_after_one, _ = sgd_step(params, first_batch)
params_after_one, loss_after_one = block_tree((params_after_one, loss_after_one))
print(f"loss used for one SGD update: {float(loss_after_one):.4f}")
