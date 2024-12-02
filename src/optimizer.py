import optax
import optax.contrib

learning_rate = 0.01

schedule_fn = optax.schedules.warmup_exponential_decay_schedule(
        init_value=learning_rate,
        peak_value=learning_rate * 1.05,
        warmup_steps=10,
        transition_steps=10,
        decay_rate=0.999,
    )
optimizer = optax.chain(
    # optax.adaptive_grad_clip(1.0),
    # optax.zero_nans(),
    optax.clip_by_global_norm(1000.0),
    optax.amsgrad(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-6),
    # optax.adam(learning_rate=schedule_fn, b1=0.9, b2=0.99, eps=1e-3, eps_root=1e-8),
    # optax.adamw(learning_rate=schedule_fn),
    # optax.ema(decay=0.999, debias=False),
)
# optimizer = optax.adamw(learning_rate=learning_rate)
transform = optax.contrib.reduce_on_plateau(
    patience=5,
    cooldown=5,
    factor=0.99,
    rtol=1e-4,
    accumulation_size=5,
    min_scale=0.01,
)
