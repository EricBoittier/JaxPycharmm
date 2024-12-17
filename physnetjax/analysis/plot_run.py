import seaborn as sns
import matplotlib.pyplot as plt


def plot_run(base_df, ax, hue, label, log=False):
    base_df = base_df[::100]
    # base_df = base_df.to_pandas()
    # Define all the metrics to plot
    metrics = [
        "train_loss", "valid_loss",
        "train_energy_mae", "valid_energy_mae",
        "train_forces_mae", "valid_forces_mae",
        "lr"
    ]

    # Plot each metric
    for i, ycol in enumerate(metrics):
        row = i % 2
        col = i // 2
        line = sns.lineplot(
            data=base_df,
            x="epoch", y=ycol,
            # hue=hue/33,
            color=sns.color_palette("Set2", 34)[hue],
            # style="f", size="nit",
            ax=ax[row][col],
            # palette="set2",
            label=label
            # legend=False
        )
        ax[row][col].legend()
        lines, labels = [], []
        # Capture lines and labels for the shared legend
        for line_obj in line.get_lines():
            lines.append(line_obj)
        labels.append(i)

        # Apply shared settings
        #
        # ax[row][col].set_xlim(1000)
        if ycol != "lr":
            ymin = base_df[ycol].min() * 0.5
            ymax = base_df[ycol].median()
            std = base_df[ycol].std()
            if isinstance(ymin, float) and isinstance(ymax, float) and isinstance(std, float):
                ax[row][col].set_ylim(ymin, ymax + std)
            elif isinstance(ymin, float) and isinstance(ymax, float):
                ax[row][col].set_ylim(ymin, ymax)
            elif isinstance(ymax, float) and isinstance(std, float):
                ax[row][col].set_ylim(0.0, ymax + std)
            else:
                # do nothing
                pass
        if log:
            ax[row][col].set_yscale("log")
        ax[row][col].set_xlabel("Epoch")
        ax[row][col].set_ylabel(ycol)
        ax[row][col].get_legend().remove()  # Remove legend from the main plot

    # Adjust the legend on the separate axis
    handles, labels = ax[row][col].get_legend_handles_labels()
    ax[-1][-1].legend(
        handles=handles, labels=labels,
        loc='center', title="Metrics"
    )
    ax[-1][-1].axis('off')  # Turn off axis for the legend space

    # plt.tight_layout()
    # plt.show()
    return ax


if __name__ == "__main__":
    from physnetjax.logging.tensorboard_interface import process_tensorboard_logs
    import polars as pl

    logs_path = ("/pchem-data/meuwly/boittier/home/pycharmm_test/"
                 "ckpts/test-ec04d45c-33e4-415e-987a-eb3548ca0770/"
                 "tfevents/")
    df = process_tensorboard_logs(logs_path)
    print(df)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plot_run(df, ax, 0, "test")
    plt.savefig("test.png")
#
# import altair as alt
#
# columns = ['valid_energy_mae', 'valid_forces_mae', 'train_energy_mae',
#            'train_forces_mae', 'train_loss', 'valid_loss', 'lr', 'batch_size',
#            'energy_w', 'charges_w', 'dipole_w', 'forces_w', 'epoch',]
#
#
# (
#     alt.Chart(base_df).mark_point(tooltip=True).encode(
#         x="epoch",
#         y="valid_forces_mae",
#         # color="species",
#     )
#     .properties(width=500)
#     .configure_scale(zero=False)
# )
#
# import polars as pl
# import polars.selectors as cs
#
# # Perform unpivoting
# df = base_df.melt(
#     id_vars=["epoch"],  # Columns to keep as identifiers
#     value_vars=columns,  # Columns to unpivot
#     variable_name="category",  # Optional renaming
#     value_name="y",  # Optional renaming
# )
#
# print(df)
#
# scaled_df = df.with_columns(
#     [
#         ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()))
#         .clip(0, 0.10)  # Apply clipping to the scaled values
#         .alias(col)        for col in df.select(cs.numeric()).columns if col != "epoch"
#     ]
# )
#
# print(scaled_df)
# cs.numeric()
#
# scaled_df = df
