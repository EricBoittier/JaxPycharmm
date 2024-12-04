import optax
import optax.contrib

base_learning_rate = 0.001

base_schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
    init_value=base_learning_rate,
    peak_value=base_learning_rate * 1.05,
    warmup_steps=10,
    transition_steps=10,
    decay_rate=0.999,
)
base_optimizer = optax.chain(
    #    optax.adaptive_grad_clip(1.0),
    optax.clip_by_global_norm(1.0),
    optax.amsgrad(learning_rate=base_schedule_fn, b1=0.9, b2=0.99, eps=1e-6),
)

base_transform = optax.contrib.reduce_on_plateau(
    patience=5,
    cooldown=5,
    factor=0.99,
    rtol=1e-4,
    accumulation_size=5,
    min_scale=0.01,
)


def get_optimizer(
    learning_rate: float = 0.01,
    schedule_fn: optax.Schedule | str | None = None,
    optimizer: optax.GradientTransformation | str | None = None,
    transform: optax.GradientTransformation | str | None = None,
    clip_global: bool | float = True,
    start_step: int = 0,
    **kwargs,
):
    if schedule_fn is None:
        schedule_fn = optax.schedules.constant_schedule(learning_rate)
    if isinstance(schedule_fn, str):
        if schedule_fn == "warmup":
            schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
                init_value=learning_rate,
                peak_value=learning_rate * 1.05,
                warmup_steps=10,
                transition_steps=10,
                decay_rate=0.999,
            )
        elif schedule_fn == "cosine_annealing":
            schedule_fn = cycled_cosine_annealing_schedule(
                init_lr=learning_rate,
                min_lr=learning_rate * 0.1,
                cycle_length=100,
                num_cycles=10,
                start_step=start_step,
            )
        elif schedule_fn == "exponential":
            schedule_fn = optax.schedules.exponential_decay_schedule(
                init_value=learning_rate, decay_rate=0.999
            )
        elif schedule_fn == "polynomial":
            schedule_fn = optax.schedules.polynomial_decay_schedule(
                init_value=learning_rate, end_value=0.0, power=1.0, transition_steps=100
            )
        elif schedule_fn == "cosine":
            schedule_fn = optax.schedules.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=100
            )
        elif schedule_fn == "constant":
            schedule_fn = optax.schedules.constant_schedule(learning_rate)
        else:
            raise ValueError(f"Unknown schedule_fn: {schedule_fn}")

    if optimizer is None:
        optimizer = optax.chain(
            # optax.adaptive_grad_clip(1.0),
            optax.clip_by_global_norm(1000.0),
            optax.amsgrad(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-6),
        )
    elif isinstance(optimizer, str):
        _chain = []
        if isinstance(clip_global, float):
            _chain.append(optax.clip_by_global_norm(clip_global))
        if optimizer == "adam":
            _chain.append(optax.adam(learning_rate=schedule_fn))
        elif optimizer == "adamw":
            _chain.append(optax.adamw(learning_rate=schedule_fn))
        elif optimizer == "amsgrad":
            _chain.append(optax.amsgrad(learning_rate=schedule_fn))

        else:
            pass
        optimizer = optax.chain(*_chain)

    if transform is None:
        transform = optax.contrib.reduce_on_plateau(
            patience=5,
            cooldown=5,
            factor=0.99,
            rtol=1e-4,
            accumulation_size=5,
            min_scale=0.01,
        )

    return optimizer, transform, schedule_fn


def cycled_cosine_annealing_schedule(
    init_lr, min_lr, cycle_length, num_cycles, start_step=0
):
    """
    Creates a cosine annealing learning rate schedule with repeated cycles.

    Args:
        init_lr (float): Initial learning rate at the start of each cycle.
        min_lr (float): Minimum learning rate after annealing within a cycle.
        cycle_length (int): Number of steps per cycle.
        num_cycles (int): Total number of cycles.
        start_step (int): Step at which the schedule starts (default is 0).

    Returns:
        optax.Schedule: A cycled cosine annealing learning rate schedule.
    """

    def schedule_fn(step):
        # Adjust step to account for the starting step
        adjusted_step = step - start_step
        if adjusted_step < 0:
            return (
                init_lr  # Return the initial learning rate until start_step is reached
            )

        # Determine the current cycle
        cycle_idx = adjusted_step // cycle_length
        if cycle_idx >= num_cycles:
            return min_lr  # Return min_lr after all cycles are complete

        # Calculate progress within the current cycle
        cycle_progress = (adjusted_step % cycle_length) / cycle_length
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * cycle_progress))
        lr = min_lr + (init_lr - min_lr) * cosine_decay
        return lr

    return schedule_fn
