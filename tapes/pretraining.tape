task prunable_model : chkpt_utils
< original=$model@bert_model
> model
:: pyenv=@ {
  python $chkpt_utils/make_pretrain_chkpt_prunable.py $original $model
}

task burn_in : bert
< in_model=$model@prunable_model
< train=$train@train_test_split
< dev=$dev@train_test_split
> out_model
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_titan .action_flags=@ {

    params="--train_batch_size 32
    --max_seq_length 128
    --max_predictions_per_seq 20
    --num_warmup_steps 10
    --keep_checkpoint_max 1
    --learning_rate 2e-5
    --output_dir $out_model
    --bert_config_file $in_model/bert_config.json
    --init_checkpoint $in_model/bert_model.ckpt"

   # Disable gradient checkpointing b/c it doesn't work for pre-training
   export DISABLE_GRAD_CHECKPOINT=True

    # Burn in the model first, without pruning
    for step in $(seq 1 3); do
      python $bert/run_pretraining.py --do_train=True --num_train_steps=$(expr $step \* 5000) --input_file=$train/* $params
      python $bert/run_pretraining.py --do_eval=True --max_eval_steps=2000 --input_file=$dev/* $params
    done
}

task prune : coreset_utils
< in_model=$out_model@burn_in
> out_model
:: sparsity=(Sparsity: 0 10 20 30 40 50 60 70 80 90)
:: prune_type=(PruneType: uniform topk)
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_16g .action_flags=@ {

    python $coreset_utils/prune_multihead_attn.py $in_model $out_model $prune_type 0.$sparsity

}

task post_prune_dev_loss : bert 
< in_model=$out_model@prune
< orig_model=$model@bert_model
< dev=$dev@train_test_split
> out
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_titan .action_flags=@ {
    params="--train_batch_size 32
    --output_dir tmp
    --max_seq_length 128
    --max_predictions_per_seq 20
    --num_warmup_steps 10
    --keep_checkpoint_max 1
    --learning_rate 2e-5
    --bert_config_file $orig_model/bert_config.json
    --init_checkpoint $in_model"

    export DISABLE_GRAD_CHECKPOINT=True
    python $bert/run_pretraining.py --do_eval=True --max_eval_steps=2000 --input_file=$dev/* $params 
    touch $out
}

task pretrain : bert 
< in_model=$out_model@prune
< orig_model=$model@bert_model
< train=$train@train_test_split
< dev=$dev@train_test_split
> out_model
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_titan .action_flags=@ {

    params="--train_batch_size 32
    --max_seq_length 128
    --max_predictions_per_seq 20
    --num_warmup_steps 10
    --keep_checkpoint_max 1
    --learning_rate 2e-5
    --output_dir $out_model
    --bert_config_file $orig_model/bert_config.json
    --init_checkpoint $in_model"

   # Disable gradient checkpointing b/c it doesn't work for pre-training
   export DISABLE_GRAD_CHECKPOINT=True

    for step in $(seq 1 10); do
        python $bert/run_pretraining.py --do_train=True --num_train_steps=$(expr $step \* 5000) --input_file=$train/* $params 
        # Note: max_eval_steps is the number of batches we process
        # default batch size is 8
        python $bert/run_pretraining.py --do_eval=True --max_eval_steps=2000 --input_file=$dev/* $params
   done
}

task approx_error : bert : scripts : coreset_utils
< pruned_model=$out_model@prune
< pre_prune_model=$out_model@burn_in
< orig_model=$model@bert_model
< sentencized=$out@sentencize[Chunk:0]
> out
:: pyenv=@ .submitter=@ .resource_flags=$resource_flags_2080 .action_flags=@ {

      # The sentencized chunk is huge, just use ~1.5k random lines
      ln $sentencized sentencized
      python $scripts/shuffle_and_split.py sentencized 5000
      params="--input_file sentencized_0
      --bert_config_file $orig_model/bert_config.json
      --layers -1
      --vocab_file $orig_model/vocab.txt
      --do_lower_case True
      --max_seq_length 128"

      # Feature files get really big, so we're going to delete this as soon as we're done with it
      python $bert/extract_features.py --init_checkpoint $pruned_model --output_file features.json $params
      python $bert/extract_features.py --init_checkpoint $pre_prune_model --output_file pre_prune_features.json $params

      python $coreset_utils/compute_approx_error.py features.json pre_prune_features.json > $out

      rm features.json
      rm pre_prune_features.json
      rm sentencized_*
}