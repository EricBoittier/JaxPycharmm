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
    table.add_column("Epoch", style="bright_cyan", no_wrap=True)
    table.add_column("time", style="green")
    table.add_column("Eff. LR", style="bright_magenta")
    table.add_column("Train Loss", style="medium_orchid3")
    table.add_column("Valid Loss", style="spring_green3")
    table.add_column("Best Loss", style="dark_goldenrod")
    table.add_column("Train Energy MAE", style="medium_orchid3")
    table.add_column("Valid Energy MAE", style="spring_green3")
    table.add_column("Train Forces MAE", style="medium_orchid3")
    table.add_column("Valid Forces MAE", style="spring_green3")
    if doCharges:
        table.add_column("Train Dipoles MAE", style="medium_orchid3")
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
        ckp,
        save_time,
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
        for i in range(10, 0, -1):
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

        # Prepare charts (for example, plot valid_loss over epochs)
        # valid_loss_panel = get_panel(self.valid_losses, "Valid Loss")
        # train_loss_panel = get_panel(self.train_losses, "Train Loss")

        # make a mini table for last checkpoint and save time
        ckp_table = Table(title="Last Checkpoint")
        ckp_table.add_column("Checkpoint", style="bright_magenta", no_wrap=True)
        ckp_table.add_column("Save Time", style="red")
        ckp_table.add_row(str(ckp), save_time)

        # Combine the table and panels into one layout
        # layout = Columns(
        #     [table, Columns([valid_loss_panel, train_loss_panel, ckp_table])]
        # )
        layout = Columns([table, ckp_table])
        return layout


styles = [
    "bright_black",
    "bright_red",
    "bright_green",
    "dark_turquoise",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
    "light_goldenrod3",
    "spring_green3",
    "light_yellow3",
    "bright_cyan",
    "bright_magenta",
]


def print_dict_as_table(data, title="Dictionary"):
    table = Table(title=title)
    for i, (key, value) in enumerate(data.items()):
        table.add_column(key, style=styles[i], no_wrap=True)
    table.add_row(*[str(value) for value in data.values()])
    return table


def pretty_print_optimizer(optimizer, transform, schedule_fn, console):
    def format_function(func):
        if hasattr(func, "__name__"):
            return f"{str(func)} {func.__name__}"
        return str(func)

    opt = {
        "init": f"{format_function(optimizer.init)}",
        "update": f"{format_function(optimizer.update)}",
    }
    trans = {
        "init": f"{format_function(transform.init)}",
        "update": f"{format_function(transform.update)}",
    }
    sched = {"func": f"{format_function(schedule_fn)}"}
    table = print_dict_as_table(opt, title="Optimizer")
    table2 = print_dict_as_table(trans, title="Transform")
    table3 = print_dict_as_table(sched, title="Schedule Function")
    console.print(table)
    console.print(table2)
    console.print(table3)


def pretty_print(optimizer, transform, schedule_fn):
    def format_function(func):
        if hasattr(func, "__name__"):
            return f"{str(func)} {func.__name__}"
        return str(func)

    optimizer_details = (
        f"Optimizer:\n"
        f"  init: {format_function(optimizer.init)}\n"
        f"  update: {format_function(optimizer.update)}"
    )
    transform_details = (
        f"Transform:\n"
        f"  init: {format_function(transform.init)}\n"
        f"  update: {format_function(transform.update)}"
    )
    schedule_fn_details = f"Schedule_fn: {format_function(schedule_fn)}"

    print("\n".join([optimizer_details, transform_details, schedule_fn_details]))


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
    table.add_column("Learning Rate", style="spring_green3", no_wrap=True)
    table.add_column("Energy Weight", style="bright_magenta")
    table.add_column("Forces Weight", style="spring_green3")
    table.add_column("Dipole Weight", style="light_goldenrod3")
    table.add_column("Charges Weight", style="bright_magenta")
    table.add_column("Batch Size", style="light_goldenrod3")
    table.add_column("Num Atoms", style="light_yellow3")

    table2 = Table(title="PhysNetJax Training Data")
    table2.add_column("Restart", style="spring_green3", no_wrap=False)
    # table2.add_column("Conversion", style="red")
    # table2.add_column("Print Freq", style="blue")
    # table2.add_column("Name", style="yellow4")
    table2.add_column("Best", style="bright_magenta")
    table2.add_column("Objective", style="light_goldenrod3")
    table2.add_column("Data Keys", style="yellow4")
    # table2.add_column("Ckpt Dir", style="blue")
    # table2.add_column("Objective", style="green")
    # table2.add_column("Saving", style="light_goldenrod3")
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
        f"{str(restart)}",
        # f"{conversion}",
        # f"{print_freq}",
        # f"{name}",
        f"{best}",
        # f"{objective}",
        f"{data_keys}",
        # f"{str(ckpt_dir)}",
        f"{objective}",
        # f"Saving a restart file each time the {objective} improves.",
    )
    return table, table2