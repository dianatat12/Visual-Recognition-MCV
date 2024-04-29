import pandas as pd
import matplotlib.pyplot as plt

# Read CSV files
train_df1 = pd.read_csv(
    "/ghome/group04/georg/MCV-C5-G4/week4/lightning_logs/text2img5epochs/metrics.csv"
)
train_df2 = pd.read_csv(
    "/ghome/group04/georg/MCV-C5-G4/week4/lightning_logs/img2text5epochs/metrics.csv"
)


# Concatenate dataframes

# Plot training and validation losses
plt.plot(
    train_df1["step"],
    train_df1["train_loss"],
    label="Text2Img Training Loss",
    color="blue",
)

plt.plot(
    train_df2["step"],
    train_df2["train_loss"],
    label="Img2Text Training Loss",
    color="red",
)


# Add labels and legend
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()

# Show plot
plt.savefig("losses.png")
plt.close()

plt.plot(
    train_df1["step"],
    train_df1["val_loss"],
    label="Text2Img Validation Loss",
    color="orange",
)

plt.plot(
    train_df2["step"],
    train_df2["val_loss"],
    label="Img2Text Validation Loss",
    color="green",
)

# Add labels and legend
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()

# Show plot
plt.savefig("val_losses.png")
plt.close()
