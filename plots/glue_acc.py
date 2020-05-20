import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO


ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "glue", "summary", "glue_acc"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
table['Sparsity'] = pd.to_numeric(table['Sparsity'], errors='raise')
table['acc'] = pd.to_numeric(table['acc'], errors='coerce')
table.round(2)


for task in pd.unique(table["GlueTask"]):

    fig, axes = plt.subplots()

    axes.set_xlabel("Sparsity")
    axes.set_ylabel("Dev Acc")

    topk = table[(table["PruneType"] == "topk") & (table["GlueTask"] == task)]
    uniform = table[(table["PruneType"] == "uniform") & (table["GlueTask"] == task)]
    axes.plot(uniform["Sparsity"], uniform["acc"], label="Uniform")
    axes.plot(topk["Sparsity"], topk["acc"], label="Top-K")
    axes.set_title(f"{task} Sparsity vs Acc")
    axes.legend()
    fig.savefig(f'plots_out/sparsity_vs_acc_{task}.png')
