import json
import pathlib
import re

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt


def extract_index(path: pathlib.Path) -> int:
    match = re.match(r"(\d+)_", path.stem)
    return int(match.group(1)) if match else -1


benchmark_dir = pathlib.Path(
    "tests/outputs/benchmark/time/Linux-CPython-3.12-64bit"
)
output_file = pathlib.Path("tests/outputs/benchmark/time/benchmark_gridgen.png")
files = sorted(benchmark_dir.glob("*.json"), key=extract_index)
data_rows = []

for json_file in files:
    tag = json_file.stem

    with json_file.open() as f:
        bench = json.load(f)

    for entry in bench["benchmarks"]:
        for value in entry["stats"]["data"]:
            data_rows.append(
                {"name": entry["name"], "version": tag, "execution_time": value}
            )

df = pl.DataFrame(data_rows)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x="version", y="execution_time", hue="name", data=df)
plt.xlabel("Version")
plt.ylabel("Execution Time (s)")
plt.legend('',frameon=False)
plt.title("Benchmark Distribution per Version")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_file)
