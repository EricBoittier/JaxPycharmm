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
