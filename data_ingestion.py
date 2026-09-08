TRAIN_EXAMPLES = 60_000
TEST_EXAMPLES = 10_000
BATCH_SIZE = 512
INPUT_DIM = 28 * 28
NUM_CLASSES = 10


def prepare_images(images):
    """Cast uint8 images to float32 in [0, 1] and flatten each one into a 1-D feature vector."""
    images = images.astype(np.float32) / 255.0
    return images.reshape(images.shape[0], -1)


rng = np.random.default_rng(0)
train_perm = rng.permutation(len(train_images))[:TRAIN_EXAMPLES]

x_train = prepare_images(train_images[train_perm])
y_train = train_labels[train_perm].astype(np.int32)
x_test = prepare_images(test_images[:TEST_EXAMPLES])
y_test = test_labels[:TEST_EXAMPLES].astype(np.int32)


def make_fixed_batches(x, y, batch_size):
    """Trim trailing examples that don't fill a batch, reshape, and move to device."""
    usable = (len(x) // batch_size) * batch_size
    x = x[:usable].reshape(usable // batch_size, batch_size, x.shape[-1])
    y = y[:usable].reshape(usable // batch_size, batch_size)
    return jax.device_put(jnp.asarray(x), device), jax.device_put(jnp.asarray(y), device)


x_train_batches, y_train_batches = make_fixed_batches(x_train, y_train, BATCH_SIZE)
x_test_batches, y_test_batches = make_fixed_batches(x_test, y_test, BATCH_SIZE)
first_batch = (x_train_batches[0], y_train_batches[0])

show_table(
    ["Array", "Shape", "Dtype", "Devices"],
    [
        ("x_train_batches", x_train_batches.shape, x_train_batches.dtype, x_train_batches.devices()),
        ("y_train_batches", y_train_batches.shape, y_train_batches.dtype, y_train_batches.devices()),
        ("x_test_batches", x_test_batches.shape, x_test_batches.dtype, x_test_batches.devices()),
        ("y_test_batches", y_test_batches.shape, y_test_batches.dtype, y_test_batches.devices()),
    ],
    title="Fixed-size batches on GPU",
)
