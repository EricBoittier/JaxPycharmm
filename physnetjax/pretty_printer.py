from rich.live import Live
from rich.table import Table
import time


def init_table(doCharges=False):
    table = Table(title="PhysNetJax Training Progress")
    table.add_column("Epoch", style="cyan", no_wrap=True)
    table.add_column("Train Loss", style="magenta")
    table.add_column("Valid Loss", style="green")
    table.add_column("Best Loss", style="red")
    table.add_column("Train Energy MAE", style="magenta")
    table.add_column("Valid Energy MAE", style="green")
    table.add_column("Train Forces MAE", style="magenta")
    table.add_column("Valid Forces MAE", style="green")
    if doCharges:
        table.add_column("Train Dipoles MAE", style="magenta")
    return table

def epoch_printer(table, epoch, train_loss, valid_loss, best_loss, train_energy_mae, valid_energy_mae,
                  train_forces_mae, valid_forces_mae, doCharges, train_dipoles_mae, valid_dipoles_mae,
                  transform_state, slr, lr_eff):
        rows = [f"{epoch: 3d}",
        f"{train_loss : 8.3f}",
        f"{valid_loss : 8.3f}",
        f"{best_loss:8.3f}",
        f"{train_energy_mae: 8.3f}",
        f"{valid_energy_mae: 8.3f}",
        f"{train_forces_mae: 8.3f}",
        f"{valid_forces_mae: 8.3f}",]
        if doCharges:
            rows.append(f"{train_dipoles_mae: 8.3f}")
        table.add_row(
            *rows
        )
        return table



def training_printer(learning_rate, energy_weight, forces_weight, dipole_weight, charges_weight, batch_size, num_atoms, restart, conversion, print_freq, name, best, objective, data_keys, ckpt_dir, train_data, valid_data):
    # new code
    table = Table(title="PhysNetJax Training Initialization")
    table.add_column("Learning Rate", style="cyan", no_wrap=True)
    table.add_column("Energy Weight", style="magenta")
    table.add_column("Forces Weight", style="green")
    table.add_column("Dipole Weight", style="red")
    table.add_column("Charges Weight", style="blue")
    table.add_column("Batch Size", style="cyan")
    table.add_column("Num Atoms", style="magenta")

    table2 = Table(title="PhysNetJax Training Initialization")
    table2.add_column("Restart", style="green")
    table2.add_column("Conversion", style="red")
    table2.add_column("Print Freq", style="blue")
    table2.add_column("Name", style="cyan")
    table2.add_column("Best", style="magenta")
    table2.add_column("Objective", style="green")
    table2.add_column("Data Keys", style="red")
    table2.add_column("Ckpt Dir", style="blue")
    table2.add_column("Train Data Keys", style="cyan")
    table2.add_column("Valid Data Keys", style="magenta")
    table2.add_column("Objective", style="green")
    table2.add_column("Saving", style="red")
    table.add_row(
        f"{learning_rate}",
        f"{energy_weight}",
        f"{forces_weight}",
        f"{dipole_weight}",
        f"{charges_weight}",
        f"{batch_size}",
        f"{num_atoms}",
    )
    table2.add_row(
        f"{restart}",
        f"{conversion}",
        f"{print_freq}",
        f"{name}",
        f"{best}",
        f"{objective}",
        f"{data_keys}",
        f"{ckpt_dir}",
        f"{train_data.keys()}",
        f"{valid_data.keys()}",
        f"{objective}",
        f"Saving a restart file each time the {objective} improves."
    )
    return table, table2
