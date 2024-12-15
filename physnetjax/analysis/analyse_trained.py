import matplotlib.pyplot as plt
import numpy as np

from physnetjax.analysis.analysis import count_params, plot_stats
from physnetjax.restart.restart import get_params_model, get_last


def save_pickle(params, location):
    import pickle

    with open(location, "wb") as f:
        pickle.dump(params, f)


def analyse_trained(
    restart, data, natoms, batch_size=100, do_plot=False, save_pickle=False
):
    params, model = get_params_model(restart)
    restart_dir = restart.parent

    if save_pickle:
        print(f"Saving pickle to {restart_dir / 'params'}.pkl")
        save_pickle(params, restart_dir / "params.pkl")

    model.natoms = natoms
    model.zbl = False
    total_params = count_params(params)
    output = plot_stats(
        data,
        model,
        params,
        _set=f"$\\ell$: {model.max_degree} | mp: {model.num_iterations} | feat.: {model.features} \n params: {total_params:.2e} | Test",
        do_kde=True,
        batch_size=batch_size,
        do_plot=do_plot,
    )
    if do_plot:
        plt.savefig(f"analysis/{str(restart_dir.name)}_test.pdf", bbox_inches="tight")
        plt.show()
    return output


if __name__ == "__main__":
    from pathlib import Path

    import jax

    from physnetjax.data.data import prepare_batches, prepare_datasets

    data_key, train_key = jax.random.split(jax.random.PRNGKey(43), 2)

    NATOMS = 37
    files = ["/pchem-data/meuwly/boittier/home/jaxeq/notebooks/ala-esp-dip-0.npz"]
    train_data, valid_data = prepare_datasets(
        data_key, 8000, 1786, files, clip_esp=False, natoms=NATOMS, clean=False
    )
    restart = (
        Path("/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/")
        / "cf3all-d069b2ca-0c5a-4fcd-b597-f8b28933693a"
    )
    restart = get_last(restart)
    print(restart)

    data_keys = ("R", "Z", "F", "E", "D", "N", "dst_idx", "src_idx", "batch_segments")
    ntest = len(valid_data["E"]) // 2
    print(ntest)
    test_data = {k: v[ntest:] for k, v in valid_data.items()}
    valid_data = {k: v[:ntest] for k, v in valid_data.items()}
    batch_size = 47

    test_batches = prepare_batches(
        data_key, test_data, batch_size, num_atoms=NATOMS, data_keys=data_keys
    )

    train_batches = prepare_batches(
        data_key, train_data, batch_size, num_atoms=NATOMS, data_keys=data_keys
    )

    valid_batches = prepare_batches(
        data_key, valid_data, batch_size, num_atoms=NATOMS, data_keys=data_keys
    )

    combined = test_batches  # + valid_batches #+ train_batches
    len(combined)

    analyse_trained(
        restart,
        combined,
        NATOMS,
        batch_size=batch_size,
        do_plot=False,
        save_pickle=True,
    )
