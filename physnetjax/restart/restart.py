from pathlib import Path
from typing import Tuple
from physnetjax.utils.utils import get_last, get_params_model

import orbax

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def restart_training(restart: str, transform, optimizer, num_atoms: int):
    """
    Restart training from a previous checkpoint.

    Args:
        restart (str): Path to the checkpoint directory
        num_atoms (int): Number of atoms in the system
    """
    restart = get_last(restart)
    _, _model = get_params_model(restart, num_atoms)
    if _model is not None:
        model = _model

    restored = orbax_checkpointer.restore(restart)
    print("Restoring from", restart)
    print("Restored keys:", restored.keys())
    state = restored["model"]
    # print(state)
    params = _  # restored["params"]
    ema_params = restored["ema_params"]
    # opt_state = restored["opt_state"]
    # print("Opt state", opt_state)
    transform_state = transform.init(params)
    # transform_state = restored["transform_state"]
    # Validate and reinitialize states if necessary
    opt_state = optimizer.init(params)
    # update mu
    # o_a, o_b = opt_state
    # from optax import ScaleByAmsgradState
    # _ = ScaleByAmsgradState(
    #     mu=opt_state[1][0]["mu"],
    #     nu=opt_state[1][0]["nu"],
    #     nu_max=opt_state[1][0]["nu_max"],
    #     count=opt_state[1][0]["count"],
    # )
    # # opt_state = (o_a, (_, o_b[1]))
    # Set training variables
    step = restored["epoch"] + 1
    best_loss = restored["best_loss"]
    print(f"Training resumed from step {step - 1}, best_loss {best_loss}")
    CKPT_DIR = Path(restart).parent
    return (
        ema_params,
        model,
        opt_state,
        params,
        transform_state,
        step,
        best_loss,
        CKPT_DIR,
        state,
    )