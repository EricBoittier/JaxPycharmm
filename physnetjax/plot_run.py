import matplotlib.pyplot as plt
import seaborn as sns


def plot_run(base_df):
    for ycol in ["train_loss", "valid_loss"]:
        sns.lineplot(base_df, x="epoch", y=ycol, label=ycol)
    plt.ylim(0, 7)
    plt.legend()
    plt.show()

    for ycol in ["train_energy_mae", "valid_energy_mae"]:
        sns.lineplot(base_df, x="epoch", y=ycol)
    plt.ylim(0, 5)
    plt.axhline(0.01749)

    plt.show()


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
