import optax
import optax.contrib
import jax.numpy as jnp

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
    optax.amsgrad(learning_rate=base_schedule_fn, b1=0.9, b2=0.99, eps=1e-3),
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
                peak_value=learning_rate * 3,
                warmup_steps=100,
                transition_steps=10,
                decay_rate=0.9999,
            )
        elif schedule_fn == "cosine_annealing":
            schedule_fn = cycled_cosine_annealing_schedule(
                init_lr=learning_rate,
            )
        elif schedule_fn == "exponential":
            schedule_fn = optax.schedules.exponential_decay_schedule(
                init_value=learning_rate, decay_rate=0.995
            )
        elif schedule_fn == "polynomial":
            schedule_fn = optax.schedules.polynomial_decay_schedule(
                init_value=learning_rate, end_value=0.0, power=1.0, transition_steps=100
            )
        elif schedule_fn == "cosine":
            schedule_fn = optax.schedules.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=5000, alpha=0.3
            )
        elif schedule_fn == "constant":
            schedule_fn = optax.schedules.constant_schedule(learning_rate)
        else:
            raise ValueError(f"Unknown schedule_fn: {schedule_fn}")

    if optimizer is None:
        optimizer = optax.chain(
            # optax.adaptive_grad_clip(1.0),
            optax.clip_by_global_norm(1.0),
            optax.amsgrad(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-3),
        )
    elif isinstance(optimizer, str):
        _chain = []
        if clip_global:
            if not isinstance(clip_global, float):
                clip_global = 1.0
            _chain.append(optax.clip_by_global_norm(clip_global))
        if optimizer == "adam":
            _chain.append(optax.adam(learning_rate=schedule_fn))
        elif optimizer == "adamw":
            _chain.append(optax.adamw(learning_rate=schedule_fn))
        elif optimizer == "amsgrad":
            _chain.append(optax.amsgrad(learning_rate=schedule_fn,  b1=0.9, b2=0.99, eps=1e-3))

        else:
            pass
        optimizer = optax.chain(*_chain)

    if transform is None:
        transform = optax.contrib.reduce_on_plateau(
            patience=5,
            cooldown=5,
            factor=0.9,
            rtol=1e-4,
            accumulation_size=5,
            min_scale=0.01,
        )

    return optimizer, transform, schedule_fn


def cycled_cosine_annealing_schedule(init_lr, period=200):
    """
    Creates a cosine annealing learning rate schedule with repeated cycles.

    Args:
        init_lr (float): Initial learning rate at the start of each cycle.
        period (int): The number of steps in each cycle.
    Returns:
        optax.Schedule: A cycled cosine annealing learning rate schedule.
    """

    # Adjust step to account for the starting step
    num_cycles = 200
    print(period, num_cycles)
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.cosine_onecycle_schedule(
                transition_steps=period // 2,
                peak_value=init_lr * (0.99**i),
                div_factor=1.3,
                final_div_factor=1.6,
            )
            for i in range(num_cycles)
        ],
        boundaries=jnp.cumsum(jnp.array([period] * num_cycles)),
    )

    return lr_schedule
