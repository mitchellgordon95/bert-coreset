import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO


ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "approx_error", "summary", "approx_error"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")

fig, axes = plt.subplots()

axes.set_xlabel("Sparsity")
axes.set_ylabel(r"$\sum |x-y|$")

topk = table[(table["PruneType"] == "topk")]
uniform = table[(table["PruneType"] == "uniform")]
axes.plot(uniform["Sparsity"], uniform["avg_error"], label="Uniform")
axes.plot(topk["Sparsity"], topk["avg_error"], label="Top-K")
axes.set_title(f"BERT Approximation Error")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_approx_err.png')
