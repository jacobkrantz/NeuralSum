
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry


@registry.register_symbol_modality("summary")
class SummaryModality(modalities.SymbolModality):
    """
    Uses defaults from SymbolModality. Overrides the loss
    function for future loss adjustment.
    """

    def loss(self, top_out, targets):
        """
        Compute loss numerator and denominator for one shard of output.
        Currently just copied from modality.Modality
        """
        logits = top_out

        # logits = tf.Print(logits, [logits])
        # targets = tf.Print(targets, [targets])

        return common_layers.padded_cross_entropy(
            logits,
            targets,
            self._model_hparams.label_smoothing,
            weights_fn=self.targets_weights_fn
        )
