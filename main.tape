import "tapes/submitters.tape"
import "tapes/versioners.tape"
import "tapes/link_data.tape"
import "tapes/data_prep.tape"
import "tapes/pretraining.tape"
import "tapes/downstream.tape"

plan glue {
  reach train_glue via (Sparsity: 0 30 60 90) * (PruneType: *) * (GlueTask: *)
}

plan approx_error {
  reach approx_error via (Sparsity: 0 30 60 90) * (PruneType: *) * (GlueTask: *)
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}