def epoch_printer(epoch, train_loss, valid_loss, best_loss, train_energy_mae, valid_energy_mae,
                  train_forces_mae, valid_forces_mae, doCharges, train_dipoles_mae, valid_dipoles_mae,
                  transform_state, schedule_fn, lr_eff):


        # Print progress.
        print(f"epoch: {epoch: 3d}\t\t\t\t train:   valid:")
        print(
            f"    loss\t\t[a.u.]     \t{train_loss : 8.3f} {valid_loss : 8.3f} {best_loss:8.3f}"
        )
        print(
            f"    energy mae\t\t[kcal/mol]\t{train_energy_mae: 8.3f} {valid_energy_mae: 8.3f}"
        )
        print(
            f"    forces mae\t\t[kcal/mol/Å]\t{train_forces_mae: 8.3f} {valid_forces_mae: 8.3f}"
        )
        if doCharges:
            print(
                f"    dipoles mae\t\t[e Å]     \t{train_dipoles_mae: 8.3f} {valid_dipoles_mae: 8.3f}"
            )
        print(
            "scale:",
            f"{transform_state.scale:.6f} {schedule_fn(epoch):.6f} LR={lr_eff:.9f}",
        )