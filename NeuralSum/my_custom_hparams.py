
from tensor2tensor.utils import registry
import tensorflow as tf

"""
Define all custom hyper parameters sets in this file. Any of these can be
    overriden in calling scripts (sh or py).
Shell override- add this flag: to t2t-traner:
    -- hparams='batch_size=256'

"""

@registry.register_hparams('exp_6')
# use this for running experiments 0-6: latest in server folder base_1.
def exp_6():
    hparams = tf.contrib.training.HParams(
        # If the problem consists of variable-length sequences
        # (see problem.batch_size_means_tokens()), then this is the number
        # of tokens per batch per GPU or per TPU core.  Otherwise, this is
        # the number of examples per GPU or per TPU core.
        batch_size=512,
        # If True, then if the features are of variable length, the batch_size is
        # used as the actual batch size (and not tokens per batch).
        use_fixed_batch_size=False,
        num_hidden_layers=6,
        kernel_height=3,
        kernel_width=1,
        hidden_size=512,
        compress_steps=0,
        # All hyperparameters ending in "dropout" are automatically set to 0.0
        # when not in training mode.
        dropout=0.2,
        clip_grad_norm=0., # i.e. no gradient clipping
        grad_noise_scale=0.0,
        summarize_grads=False,
        # Whether to log the name and size of every variable
        summarize_vars=False,
        initializer="uniform_unit_scaling", # or orthogonal, maybe others
        initializer_gain=1.0,
        label_smoothing=0.1,
        optimizer="Adam",
        optimizer_adam_epsilon=1e-9,
        optimizer_adam_beta1=0.90,
        optimizer_adam_beta2=0.98,
        optimizer_momentum_momentum=0.9,
        optimizer_momentum_nesterov=False,
        optimizer_adafactor_beta1=0.0,
        optimizer_adafactor_beta2=0.999,
        optimizer_adafactor_factored=True,
        optimizer_adafactor_decay_type="pow",
        optimizer_adafactor_memory_exponent=0.8,
        optimizer_adafactor_clipping_threshold=1.0,
        optimizer_adafactor_multiply_by_parameter_scale=True,
        # Number of accumulating steps for multi step optimizers.
        optimizer_multistep_accumulate_steps=None,
        weight_decay=0.0, # 1e-6 for some decay
        weight_noise=0.0,
        # Defines the learning rate as a product of named functions.
        # Available functions are listed in learning_rate._LEARNING_RATE_FUNCTIONS
        # e.g. "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"
        learning_rate_schedule="legacy",
        learning_rate_constant=1.0,
        # If learning_rate_schedule=="legacy",
        # then we specify decay scheme here.  Warmup is always exponential,
        # except with "noam" learning rate decay scheme.
        # see optimize.legacy_learning_rate_schedule()
        learning_rate_decay_scheme="noam", # could be "none"
        # decay_steps and decay_staircase for learning_rate_decay_scheme=="exp"
        learning_rate_decay_steps=5000,
        learning_rate_decay_staircase=False,
        learning_rate_minimum=None,
        learning_rate_decay_rate=1.0,
        learning_rate_warmup_steps=8000,
        learning_rate_cosine_cycle_steps=250000,
        learning_rate=0.2,
        sampling_method="argmax",  # "argmax" or "random"
        sampling_temp=1.0,  # temperature for sampling
        # expand the logits a piece at a time - saves memory.
        factored_logits=False,
        multiply_embedding_mode="sqrt_depth",
        # Parameters related to mixtures of experts.
        moe_hidden_sizes="2048",  # hidden layer sizes (comma-separated)
        moe_num_experts=64,  # number of experts per layer
        moe_k=2,  # how many experts to use for each batch element
        moe_loss_coef=1e-2,
        # Sequences of operations to perform on layer input and layer output.
        # Used by common_layers.layer_preprocess, common_layers.layer_postprocess
        # Each character represents an operation:
        # none: no preprocessing
        #    d: apply dropout
        #    n: apply normalization (see norm_type and norm_epsilon)
        #    a: add layer input (residual connection - only during postprocess)
        # The special string "none" is used instead of the empty string
        # to indicate no pre/postprocessing, since the empty string causes
        # trouble for hyperparameter tuning.
        # TODO(noam): The current settings ("", "dan") are the published version
        # of the transformer.  ("n", "da") seems better for harder-to-learn
        # models, so it should probably be the default.
        layer_preprocess_sequence="none",
        layer_postprocess_sequence="dan",
        # dropout rate to use during layer_preprocess and layer_postprocess
        layer_prepostprocess_dropout=0.1,
        # broadcast dimensions for layer_prepostprocess_dropout
        # a comma-separated list of integers.
        # see common_layers.dropout_with_broadcast_dims()
        # Change this to "1" to save memory.
        layer_prepostprocess_dropout_broadcast_dims="",
        # dropout some symbols (set them to 0) before embedding.
        symbol_dropout=0.0,
        # What type of normalization to use
        norm_type="layer",  # "batch", layer", "noam", "none".
        # epsilon parameter to normalization function
        norm_epsilon=1e-6,
        symbol_modality_num_shards=16,
        # During training, we drop sequences whose inputs and targets are shorter
        # than min_length
        min_length=0,
        # During training, we drop sequences whose inputs or targets are longer
        # than max_length.
        # If max_length==0, we use hparams.batch_size instead.
        max_length=0,
        # Maximum length in the smallest length bucket.  Setting this
        # flag too high will result in wasteful padding of short
        # sequences.  Due to some (hopefully) temporary hacks in the
        # data reading and batching code, setting this flag too low
        # results in a very long batch-shuffling queue.
        min_length_bucket=8,
        # This flag controls the number of length buckets in the data
        # reader.  The buckets have maximum lengths from
        # min_bucket_length to (max_length or batch_size), increasing
        # (approximately) by factors of length_bucket_step.
        length_bucket_step=1.1,
        # If set to True, drop sequences longer than max_length during eval.
        # This affects the validity of the evaluation metrics.
        eval_drop_long_sequences=False,
        # If True, run the model autoregressively instead of teacher-forcing
        # during eval
        eval_run_autoregressive=False,
        # You can also share the input embeddings with the output embeddings
        # by using a problem_hparams that uses the same modality object for
        # the input_modality and target_modality.
        shared_embedding_and_softmax_weights=True,
        # In SymbolModality, skip the top layer, assume we're providing logits.
        symbol_modality_skip_top=False,
        # For each feature for which you want to override the default input
        # modality, add an entry to this semicolon-separated string. Entries are
        # formatted "feature_name:modality_type:modality_name", e.g.
        # "inputs:symbol:default;other_inputs:audio:identity".
        input_modalities="symbol:summary",  # We don't use empty string in params.
        # To override the default target modality, specify
        # "modality_type:modality_name", e.g. "symbol:ctc".
        target_modality="symbol:summary",
        # @jacobkrantz: input_modalities and target_modality were "default".
        # The maximum length of "input" sequence.
        # Sequences longer than this value will be truncated. 0 or negative values
        # mean there is no maximum or truncation.
        # You can change this behavior by overriding preprocess_example() method
        # in your problem class.
        max_input_seq_length=0,
        # The maximum length of "target" sequence.
        # Sequences longer than this value will be truncated. 0 or negative values
        # mean there is no maximum or truncation.
        # You can change this behavior by overriding preprocess_example() method
        # in your problem class.
        max_target_seq_length=0,
        # if nonzero, we split the target sequences on example read.
        # This is for use with language modeling problems with fixed length
        # examples.  e.g.  The examples may be written with length 65536, but we
        # want to split each example into 64 examples of length 1024.
        split_to_length=0,
        # Video settings: how many frames to batch on input and targets.
        video_num_input_frames=1,
        video_num_target_frames=1,
        # This flag allows us to optionally treat a seq-to-seq problem
        # as a language model.  Legal values are:
        #
        # "none" - Do not prepend the inputs to the targets.
        # "prepend_inputs_masked_attention"
        #     replace "targets" in preprocessing with
        #     tf.concat([inputs, [0], targets], axis=1)
        #     i.e. we prepend the inputs to the targets with a single
        #     padding token in between.  Use masked self-attention on the
        #     entire resulting sequence.  During training, we compute losses on
        #     the combined sequence.  During eval, we compute the metrics
        #     on only the targets portion.
        # "prepend_inputs_full_attention"
        #     similar to the previous option except that each
        #     position in the inputs portion can see the
        #     entire inputs portion.  This removes the challenge of
        #     autoregressively predicting the inputs portion.
        # @jacobkrantz was prepend_inputs_masked_attention. It is a bad idea to
        #     prepend inputs to targets for summarization.
        prepend_mode="none",
        # Scheduled sampling is interesting for auto-regressive models.
        # It runs an additional step using the generated output as autoregressive
        # targets, which can improve the models inference results later. The
        # parameter scheduled_sampling_prob determines with what probability
        # will such additional step be run. It's turned off (0.0) by default.
        # This probability will exponentially warm up for the number of
        # steps determined by scheduled_sampling_warmup_steps.
        # The tensor used for the second step will consist of outputs from
        # the first step mixed with gold truth, with the proportion of gold
        # determined by scheduled_sampling_gold_mixin_prob.
        scheduled_sampling_prob=0.0,
        scheduled_sampling_warmup_steps=50000,
        scheduled_sampling_gold_mixin_prob=0.5,
        # This setting controls whether to copy variables around in a daisy chain
        # (if true) or leave their placement to TensorFlow. It only affects multi
        # device training and mostly should be turned on for performance. One
        # exception are recurrent models: with dynamic loops it must be off.
        daisy_chain_variables=True,
        # If True in PREDICT mode, then last-position-only optimizations are not
        # used.
        force_full_predict=False,
        # Set this for pure model parallelism.  There is only one data shard.
        no_data_parallelism=False,
        # dtype used for activations. - "float32" or "bfloat16"
        # activation_dtype="bfloat16" currently only works on TPU.
        #    It lowers activation-memory usage
        #    and does not appear to affect quality.
        #    You can train on TPU with activation_dtype="bfloat16" and evaluate
        #    on CPU/GPU with activation_dtype="float32"
        activation_dtype="float32",
        # dtype used for parameters: "float32" or "bfloat16"
        # bfloat16 currently only works with optimizer="adafactor".
        #   The savings in memory allow for training larger models.
        #   Weights are encoded as (w*128)^8, using pseudostochastic
        #   roundoff.  Initial experiments show that model quality is similar
        #   to baseline for about 3M training steps, but worse thereafter.
        weight_dtype="float32",
    )

    # adjustments were made to above params. They started as 'basic_1' and
    #    became 'transformer_base_v1'. Additions below:

    # Add new ones like this.
    hparams.add_hparam("filter_size", 2048)
    hparams.add_hparam("num_sampled_classes", 0)
    # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
    hparams.add_hparam("num_encoder_layers", 0)
    hparams.add_hparam("num_decoder_layers", 0)
    # Attention-related flags.
    hparams.add_hparam("num_heads", 8)
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("ffn_layer", "dense_relu_dense")
    hparams.add_hparam("parameter_attention_key_channels", 0)
    hparams.add_hparam("parameter_attention_value_channels", 0)
    # All hyperparameters ending in "dropout" are automatically set to 0.0
    # when not in training mode.
    hparams.add_hparam("attention_dropout", 0.1)
    hparams.add_hparam("attention_dropout_broadcast_dims", "")
    hparams.add_hparam("relu_dropout", 0.1)
    hparams.add_hparam("relu_dropout_broadcast_dims", "")
    hparams.add_hparam("pos", "timing")  # timing, none
    hparams.add_hparam("nbr_decoder_problems", 1)
    hparams.add_hparam("proximity_bias", False)
    hparams.add_hparam("causal_decoder_self_attention", True)
    hparams.add_hparam("use_pad_remover", True)
    hparams.add_hparam("self_attention_type", "dot_product") # ignored 6/26
    hparams.add_hparam("max_relative_position", 0)
    hparams.add_hparam("conv_first_kernel", 3)
    # These parameters are only used when ffn_layer=="local_moe_tpu"
    hparams.add_hparam("moe_overhead_train", 1.0)
    hparams.add_hparam("moe_overhead_eval", 2.0)
    # local self attention parameters.
    # block_width is filter width. should be less than block_length.
    hparams.add_hparam("block_length", 128)
    hparams.add_hparam("block_width", 128)
    # dilated attention parameters
    hparams.add_hparam('gap_size', 0)
    hparams.add_hparam('num_memory_blocks', 2)
    hparams.moe_num_experts = 16
    hparams.moe_loss_coef = 1e-3

    # These are added for param version transformer_prepend_v2 called by
    #    'transformer_prepend'
    hparams.layer_preprocess_sequence = "n"
    hparams.layer_postprocess_sequence = "da"

    # These params define which attention mechanism is used for each of
    #   the three multiheaded attention mechanisms.
    #   hparams.self_attention_type now does nothing.
    # Can be one of:
    #   "dot_product", "dot_product_relative", "dot_product_relative_v2",
    #   "local_mask_right", "local_within_block_mask_right", "local_unmasked",
    #   "masked_dilated_1d", "unmasked_dilated_1d", "edge_vector"
    #
    # graph attention still works this way for T2T v.1.6.5.
    # future versions moved graph attention to message passing:
    #   commit 90c5f41f92dd35b37d7e0003f81e82c8bae6648c
    hparams.add_hparam("encoder_self_attention_type", 'dot_product')
    hparams.add_hparam("decoder_self_attention_type", 'dot_product')
    hparams.add_hparam("enc_dec_attention_type", 'dot_product')

    # serve parameters as object: tf.contrib.training.HParams
    return hparams

@registry.register_hparams('exp_7')
def exp_7():
    # make small adjustments to exp_6 params.
    hparams = exp_6()
    hparams.hidden_size = 1024
    hparams.moe_hidden_sizes="4096",
    hparams.num_heads = 16
    return hparams

@registry.register_hparams('exp_11')
def exp_11():
    hparams = exp_6()
    # mess with attention mechanisms.
    # No changes are to be made (baseline).
    hparams.encoder_self_attention_type = 'dot_product'
    hparams.decoder_self_attention_type = 'dot_product'
    hparams.enc_dec_attention_type = 'dot_product'
    return hparams

@registry.register_hparams('exp_12')
def exp_12():
    hparams = exp_6()
    # mess with attention mechanisms.
    # Use relative positional embeddings instead of absolute
    # dot_product_relative cannot be used with enc_dec attention
    hparams.encoder_self_attention_type = 'dot_product_relative'
    hparams.decoder_self_attention_type = 'dot_product_relative'
    hparams.enc_dec_attention_type = 'dot_product'
    hparams.max_relative_position = 20
    hparams.pos = None
    return hparams

@registry.register_hparams('exp_13')
def exp_13():
    hparams = exp_6()
    # Bidirectional Block Self-Attention
    # https://arxiv.org/abs/1804.00857
    hparams.encoder_self_attention_type = 'local_unmasked'
    hparams.decoder_self_attention_type = 'local_unmasked'
    hparams.enc_dec_attention_type = 'dot_product'

    hparams.block_length = 128
    # filter width:
    hparams.block_width = 100 # 100 is default of local_attention_1d()
    return hparams

@registry.register_hparams('exp_14')
def exp_14():
    # increase the number of vector components to learn relative
    # embeddings for.
    hparams = exp_12()
    hparams.max_relative_position = 100
    return hparams

@registry.register_hparams('exp_15')
def exp_15():
    # increase the number of vector components to learn relative
    # embeddings for.
    hparams = exp_12()
    hparams.max_relative_position = 8
    return hparams

@registry.register_hparams('exp_16')
def exp_16():
    hparams = exp_6()
    hparams.encoder_self_attention_type = 'unmasked_dilated_1d'
    hparams.decoder_self_attention_type = 'unmasked_dilated_1d'
    hparams.enc_dec_attention_type = 'dot_product'

    hparams.block_length = 64
    hparams.block_width = 64
    hparams.gap_size = 0
    hparams.num_memory_blocks = 4
    return hparams

@registry.register_hparams('exp_17')
def exp_17():
    # same as exp_17, but mask the right side of memory blocks
    hparams = exp_16()
    hparams.encoder_self_attention_type = 'masked_dilated_1d'
    hparams.decoder_self_attention_type = 'masked_dilated_1d'
    return hparams

@registry.register_hparams('exp_18')
def exp_18():
    # split sequence into memory blocks and mask the right side.
    hparams = exp_6()
    hparams.encoder_self_attention_type = 'local_mask_right'
    hparams.decoder_self_attention_type = 'local_mask_right'
    hparams.enc_dec_attention_type = 'dot_product'

    hparams.block_length = 128
    return hparams

@registry.register_hparams('exp_19')
def exp_19():
    # Not sure what this one does.
    hparams = exp_6()
    hparams.encoder_self_attention_type = 'local_within_block_mask_right'
    hparams.decoder_self_attention_type = 'local_within_block_mask_right'
    hparams.enc_dec_attention_type = 'dot_product'

    hparams.block_length = 128
    return hparams

@registry.register_hparams('exp_20')
def exp_20():
    # mask the right side of memory blocks in the decoder. Leave the encoder
    # unmasked.
    hparams = exp_16()
    hparams.encoder_self_attention_type = 'unmasked_dilated_1d'
    hparams.decoder_self_attention_type = 'masked_dilated_1d'
    return hparams

@registry.register_hparams('exp_21')
def exp_21():
    # mask the right side of memory blocks in the ecoder. Leave the decoder
    # unmasked. Did not generate any summaries...
    hparams = exp_16()
    hparams.encoder_self_attention_type = 'masked_dilated_1d'
    hparams.decoder_self_attention_type = 'unmasked_dilated_1d'
    return hparams

# # throws an error. Needs more time to debug.
# @registry.register_hparams('exp_n') # tensor dimension error
# def exp_n():
#     # Use experiment 12, except mask the right side of positional
#     #   embeddings.
#     # Wait for exp 14 to finish so that we know which is better, 12 or 14.
#     # replace exp_12() with exp_14() if 14 performs better.
#     hparams = exp_12()
#     hparams.encoder_self_attention_type = 'dot_product_relative_v2'
#     hparams.decoder_self_attention_type = 'dot_product_relative_v2'
#     return hparams
