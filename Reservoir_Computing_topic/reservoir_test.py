from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse, rsquare
from reservoirpy.nodes import Reservoir, Ridge
from pathlib import Path
import pandas as pd


# Set the path to the downloaded data
download_path = Path.cwd() / ".dataset"

# Read labels file
labels_file = download_path / "Y_train_ofTdMHi.csv"
df = pd.read_csv(labels_file)

# Construct file path by concatenating folder and file name
df["relative_path"] = str(download_path) + "/X_train/" + df["id"]

# Drop id column (replaced it with relative_path)
df.drop(columns=["id"], inplace=True)

df.rename(columns={"pos_label": "label"}, inplace=True)

# invert relative_path and label columns positions
df = df[["relative_path", "label"]]


X = mackey_glass(n_timesteps=2000)


reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout = Ridge(output_dim=1, ridge=1e-5)


esn = reservoir >> readout


esn.fit(X[:500], X[1:501], warmup=100)

predictions = esn.run(X[501:-1])

predictions = esn.fit(X[:500], X[1:501]).run(X[501:-1])



print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))
