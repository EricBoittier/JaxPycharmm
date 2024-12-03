import optax
import optax.contrib

learning_rate = 0.01

base_schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
    init_value=learning_rate,
    peak_value=learning_rate * 1.05,
    warmup_steps=10,
    transition_steps=10,
    decay_rate=0.999,
)
base_optimizer = optax.chain(
    # optax.adaptive_grad_clip(1.0),
    # optax.zero_nans(),
    optax.clip_by_global_norm(1000.0),
    optax.amsgrad(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-6),
    # optax.adam(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-3, eps_root=1e-8),
    # optax.adamw(learning_rate=schedule_fn),
    # optax.ema(decay=0.999, debias=False),
)
# optimizer = optax.adamw(learning_rate=learning_rate)
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
            optax.adaptive_grad_clip(1.0),
            optax.clip_by_global_norm(1000.0),
            optax.amsgrad(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-6),
        )
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
