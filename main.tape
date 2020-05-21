import "tapes/submitters.tape"
import "tapes/versioners.tape"
import "tapes/link_data.tape"
import "tapes/data_prep.tape"
import "tapes/pretraining.tape"
import "tapes/downstream.tape"

plan glue {
  reach train_glue via (Sparsity: 0 30 60 90) * (PruneType: *) * (GlueTask: *)
}

# summary glue_acc {
#   of train_glue > acc {
#     tail -4 $(dirname $out_model)/job.out | head -n 1 | awk '{print $4}' > $acc
#   }
# }

summary post_finetune_dev_loss {
  of pretrain > loss {
    cat $(dirname $out_model)/job.out | tail -n 5 | head -n 1 | awk '{print $4}' > $loss
  }
}

plan post_prune_dev_loss {
  reach post_prune_dev_loss via (Sparsity: 0 30 60 90) * (PruneType: *)
}

summary post_prune_dev_loss {
  of post_prune_dev_loss > loss {
    cat $(dirname $out)/job.out | tail -n 5 | head -n 1 | awk '{print $4}' > $loss
  }
}

plan approx_error {
  reach approx_error via (Sparsity: 0 30 60 90) * (PruneType: *)
}

summary approx_error {
  of approx_error > avg_error {
    cat $out | awk '{print $5}' > $avg_error
  }
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}