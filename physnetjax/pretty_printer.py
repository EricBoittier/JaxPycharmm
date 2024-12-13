from rich.live import Live
from rich.table import Table
import time
import time
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
import asciichartpy as acp
import numpy as np
from rich.columns import Columns


def get_panel(data, title):
    return Panel(
        acp.plot(data),
        expand=False,
        title=f"~~ [bold][yellow]{title}[/bold][/yellow] ~~",
    )


def init_table(doCharges=False):
    table = Table(title="PhysNetJax Training Progress")
    table.add_column("Epoch", style="cyan", no_wrap=True)
    table.add_column("time", style="green")
    table.add_column("Eff. LR", style="magenta")
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


class Printer:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.valid_losses = []
        self.best_losses = []
        self.train_energy_maes = []
        self.valid_energy_maes = []
        self.train_forces_maes = []
        self.valid_forces_maes = []
        self.train_dipoles_maes = []
        self.valid_dipoles_maes = []
        self.transform_states = []
        self.slrs = []
        self.lr_effs = []
        self.epoch_lengths = []

    def update(
        self,
        epoch,
        train_loss,
        valid_loss,
        best_loss,
        train_energy_mae,
        valid_energy_mae,
        train_forces_mae,
        valid_forces_mae,
        doCharges,
        train_dipoles_mae,
        valid_dipoles_mae,
        transform_state,
        slr,
        lr_eff,
        epoch_length,
    ):

        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.best_losses.append(best_loss)
        self.train_energy_maes.append(train_energy_mae)
        self.valid_energy_maes.append(valid_energy_mae)
        self.train_forces_maes.append(train_forces_mae)
        self.valid_forces_maes.append(valid_forces_mae)
        self.train_dipoles_maes.append(train_dipoles_mae)
        self.valid_dipoles_maes.append(valid_dipoles_mae)
        self.transform_states.append(transform_state)
        self.slrs.append(slr)
        self.lr_effs.append(lr_eff)
        self.epoch_lengths.append(epoch_length)

        table = init_table(doCharges)
        # update the table with the last few data points
        for i in range(5, 0, -1):
            if len(self.epochs) >= i:
                table = epoch_printer(
                    table,
                    self.epochs[-i],
                    self.train_losses[-i],
                    self.valid_losses[-i],
                    self.best_losses[-i],
                    self.train_energy_maes[-i],
                    self.valid_energy_maes[-i],
                    self.train_forces_maes[-i],
                    self.valid_forces_maes[-i],
                    doCharges,
                    self.train_dipoles_maes[-i],
                    self.valid_dipoles_maes[-i],
                    self.transform_states[-i],
                    self.slrs[-i],
                    self.lr_effs[-i],
                    self.epoch_lengths[-i],
                )
        return table


def epoch_printer(
    table,
    epoch,
    train_loss,
    valid_loss,
    best_loss,
    train_energy_mae,
    valid_energy_mae,
    train_forces_mae,
    valid_forces_mae,
    doCharges,
    train_dipoles_mae,
    valid_dipoles_mae,
    transform_state,
    slr,
    lr_eff,
    epoch_length,
):
    rows = [
        f"{epoch: 3d}",
        f"{epoch_length}",
        f"{lr_eff: 8.3e}",
        f"{train_loss : 8.3f}",
        f"{valid_loss : 8.3f}",
        f"{best_loss:8.3f}",
        f"{train_energy_mae: 8.3f}",
        f"{valid_energy_mae: 8.3f}",
        f"{train_forces_mae: 8.3f}",
        f"{valid_forces_mae: 8.3f}",
    ]
    if doCharges:
        rows.append(f"{train_dipoles_mae: 8.3f}")
    table.add_row(*rows)
    return table


def training_printer(
    learning_rate,
    energy_weight,
    forces_weight,
    dipole_weight,
    charges_weight,
    batch_size,
    num_atoms,
    restart,
    conversion,
    print_freq,
    name,
    best,
    objective,
    data_keys,
    ckpt_dir,
    train_data,
    valid_data,
):
    # new code
    table = Table(title="PhysNetJax Training Params")
    table.add_column("Learning Rate", style="cyan", no_wrap=True)
    table.add_column("Energy Weight", style="magenta")
    table.add_column("Forces Weight", style="green")
    table.add_column("Dipole Weight", style="red")
    table.add_column("Charges Weight", style="blue")
    table.add_column("Batch Size", style="cyan")
    table.add_column("Num Atoms", style="magenta")

    table2 = Table(title="PhysNetJax Training Data")
    table2.add_column("Restart", style="green")
    table2.add_column("Conversion", style="red")
    table2.add_column("Print Freq", style="blue")
    table2.add_column("Name", style="cyan")
    table2.add_column("Best", style="magenta")
    table2.add_column("Objective", style="green")
    table2.add_column("Data Keys", style="red")
    table2.add_column("Ckpt Dir", style="blue")
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
        f"{objective}",
        f"Saving a restart file each time the {objective} improves.",
    )
    return table, table2
