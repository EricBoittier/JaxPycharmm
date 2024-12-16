import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import jax
from physnetjax.analysis.analysis import count_params, plot_stats
from physnetjax.restart.restart import get_params_model, get_last
from physnetjax.data.data import prepare_batches, prepare_datasets
from typing import Dict, List, Any

# --- CONFIGURATION ---
DEFAULT_BATCH_SIZE = 100
DEFAULT_NATOMS = 47
DEFAULT_PRNG = 43
DEFAULT_DATA_KEYS = (
    "R", "Z", "F", "E", "D", "N", "dst_idx", "src_idx", "batch_segments"
)


# --- UTILITY FUNCTIONS ---


def save_pickle(data: Any, location: Path) -> None:
    """Saves data as a pickle file."""
    with open(location, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {location}")


def load_restart(restart_path: Path) -> Path:
    """Loads the latest checkpoint from the provided restart path."""
    restart = get_last(restart_path)
    print(f"Loaded restart: {restart}")
    return restart


# --- DATA PREPARATION ---


def load_data(
        NATOMS: int,
        PRNG: int = DEFAULT_PRNG,
        files: List[str] = None,
        num_train: int = 8000,
        num_valid: int = 1786,
        data_keys: tuple = DEFAULT_DATA_KEYS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        load_test: bool = True,
        load_train: bool = False,
        load_validation: bool = False,
) -> Dict[str, Any]:
    """
    Prepares specified batches (test, train, validation) from dataset files.

    Args:
        NATOMS: Number of atoms in the dataset.
        PRNG: Seed for reproducibility.
        files: List of file paths containing the datasets.
        num_train: Number of training samples.
        num_valid: Number of validation samples.
        data_keys: Tuple of dataset keys.
        batch_size: The size of each batch.
        load_test: If True, loads test data.
        load_train: If True, loads train data.
        load_validation: If True, loads validation data.

    Returns:
        A dictionary of prepared data batches.
    """
    if not files:
        raise ValueError("No files provided for loading data.")

    # Generate PRNG keys
    data_key, _ = jax.random.split(jax.random.PRNGKey(PRNG), 2)

    # Prepare the datasets
    train_data, valid_data = prepare_datasets(
        data_key, num_train, num_valid, files, clip_esp=False, natoms=NATOMS, clean=False
    )
    ntest = len(valid_data["E"]) // 2

    # Split into validation and testing datasets
    test_data = {k: v[ntest:] for k, v in valid_data.items()}
    valid_data = {k: v[:ntest] for k, v in valid_data.items()}

    # Prepare batches based on requested data
    combined_batches = {}
    if load_test:
        test_batches = prepare_batches(
            data_key, test_data, batch_size, num_atoms=NATOMS, data_keys=data_keys
        )
        combined_batches["test"] = test_batches
    if load_train:
        train_batches = prepare_batches(
            data_key, train_data, batch_size, num_atoms=NATOMS, data_keys=data_keys
        )
        combined_batches["train"] = train_batches
    if load_validation:
        valid_batches = prepare_batches(
            data_key, valid_data, batch_size, num_atoms=NATOMS, data_keys=data_keys
        )
        combined_batches["validation"] = valid_batches

    print(f"Loaded data batches: {list(combined_batches.keys())}")
    return combined_batches


# --- MODEL ANALYSIS ---


def analyse_trained(
        restart: Path,
        data: Dict[str, Any],
        natoms: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        do_plot: bool = False,
        save_results: bool = False,
) -> Dict[str, Any]:
    """
    Analyse a model with trained parameters.

    Args:
        restart: Path to the restart model directory.
        data: The dataset to analyze (supports multiple keys).
        natoms: Number of atoms in the model.
        batch_size: Size of data batches.
        do_plot: Whether to generate plots.
        save_results: Whether to save results to pickle files.

    Returns:
        Output from the `plot_stats` function.
    """
    params, model = get_params_model(restart)
    restart_dir = restart.parent

    # Save parameters and model attributes if required
    if save_results:
        save_pickle(params, restart_dir / "params.pkl")
        model_kwargs = model.return_attributes()
        save_pickle(model_kwargs, restart_dir / "model_kwargs.pkl")

    # Set model attributes
    model.natoms = natoms
    model.zbl = False
    total_params = count_params(params)

    # Generate analysis and plots
    output = plot_stats(
        data,
        model,
        params,
        _set=(
            f"$\\ell$: {model.max_degree} | mp: {model.num_iterations} | "
            f"feat.: {model.features} \n params: {total_params:.2e} | Test"
        ),
        do_kde=True,
        batch_size=batch_size,
        do_plot=do_plot,
    )
    if do_plot:
        plt.savefig(f"analysis/{restart_dir.name}_test.pdf", bbox_inches="tight")
        plt.show()

    return output


# --- ARGPARSE ---

def parse_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Analyse trained PhysNet models.")
    parser.add_argument(
        "--restart",
        type=str,
        required=True,
        help="Path to the directory containing restart checkpoints.",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="File paths containing data used for training/validation/testing.",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        default=DEFAULT_NATOMS,
        help="Number of atoms in the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size to use during analysis.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=8000,
        help="Number of training samples.",
    )
    parser.add_argument(
        "--num_valid",
        type=int,
        default=1786,
        help="Number of validation samples.",
    )
    parser.add_argument(
        "--prng",
        type=int,
        default=DEFAULT_PRNG,
        help="PRNG seed for reproducibility.",
    )
    parser.add_argument(
        "--load_test",
        action="store_true",
        help="Flag to load test data.",
    )
    parser.add_argument(
        "--load_train",
        action="store_true",
        help="Flag to load train data.",
    )
    parser.add_argument(
        "--load_validation",
        action="store_true",
        help="Flag to load validation data.",
    )
    parser.add_argument(
        "--do_plot",
        action="store_true",
        help="Flag to generate and save plots.",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Flag to save analysis results as pickle files.",
    )
    return parser.parse_args()


# --- MAIN EXECUTION ---


def main():
    """
    Main entry point for the script. Handles loading of data, configuration, and analysis.
    """
    args = parse_args()
    from physnetjax.utils.pretty_printer import print_dict_as_table
    print_dict_as_table(dict(args), "args", plot=True)

    # Parse inputs from args
    restart_path = Path(args.restart)
    files = args.files
    natoms = args.natoms
    batch_size = args.batch_size
    num_train = args.num_train
    num_valid = args.num_valid

    # Load the latest restart
    restart = load_restart(restart_path)

    # Load specified data
    data = load_data(
        NATOMS=natoms,
        PRNG=args.prng,
        files=files,
        num_train=num_train,
        num_valid=num_valid,
        batch_size=batch_size,
        load_test=args.load_test,
        load_train=args.load_train,
        load_validation=args.load_validation,
    )

    # Analyze model
    combined_test_batches = data.get("test", {})
    result = analyse_trained(
        restart,
        combined_test_batches,
        natoms,
        batch_size=batch_size,
        do_plot=args.do_plot,
        save_results=args.save_results,
    )

    print("Analysis completed:", result)


if __name__ == "__main__":
    main()