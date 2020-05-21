import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from io import StringIO


ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "glue", "summary", "glue_acc"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
table['Sparsity'] = pd.to_numeric(table['Sparsity'], errors='raise')
table['acc'] = pd.to_numeric(table['acc'], errors='coerce')
table.round(2)

unique_sparsity = pd.unique(table['Sparsity'])
total_uniform_acc = np.zeros(unique_sparsity.shape[0])
total_topk_acc = np.zeros(unique_sparsity.shape[0])
count = 0
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

    if len(uniform["acc"]) == unique_sparsity.shape[0] and len(topk["acc"]) == unique_sparsity.shape[0] and not uniform["acc"].isnull().values.any() and not topk["acc"].isnull().values.any():
        total_uniform_acc += uniform["acc"].to_numpy()
        total_topk_acc += topk["acc"].to_numpy()
        count += 1
    else:
        print(f"Skipping {task} for averages since it's not complete")

fig, axes = plt.subplots()

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev Acc")
axes.plot(unique_sparsity, total_uniform_acc / count, label="Uniform")
axes.plot(unique_sparsity, total_topk_acc / count, label="Top-K")
axes.set_title(f"Average GLUE Sparsity vs Acc")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_avg.png')
