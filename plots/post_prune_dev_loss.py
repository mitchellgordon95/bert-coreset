import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO


ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "post_prune_dev_loss", "summary", "post_prune_dev_loss"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
table['Sparsity'] = pd.to_numeric(table['Sparsity'], errors='raise')
table['loss'] = pd.to_numeric(table['loss'], errors='coerce')
table.round(2)

fig, axes = plt.subplots()

axes.set_xlabel("Sparsity")
axes.set_ylabel(r"Dev Loss")

topk = table[(table["PruneType"] == "topk")]
uniform = table[(table["PruneType"] == "uniform")]
axes.plot(uniform["Sparsity"], uniform["loss"], label="Uniform")
axes.plot(topk["Sparsity"], topk["loss"], label="Top-K")
axes.set_title(f"BERT Post-Prune Dev Loss")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_pretrain_dev_loss.png')
