
import tensorflow as tf

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry


@registry.register_symbol_modality("summary")
class SummaryModality(modalities.SymbolModality):
    """
    Uses defaults from SymbolModality. Overrides the loss
    function for future loss adjustment.
    """

    def loss(self, logits, inputs, targets):
        """
        Compute loss numerator and denominator for one shard of output.
            different function definition than modality.Modality. Currently must
            be used with MyT2TModel.
            logits come directly from model.top(outputs, features)
        Args:
            logits: tensor of shape [batch, timesteps, 1,1, vocab_size]
            inputs: tensor of shape [batch, timesteps, 1,1]
            targets: tensor of shape [batch, timesteps, 1,1]
            (1,1 is optional)
        Returns:
            loss_numerator: a `Scalar`.  Sum of losses.
            loss_denominator: a `Scalar.  The number of non-padding target tokens.
        """

        # # prints for dynamic tensor contents:
        # logits = tf.Print(logits, [logits],
        #     message='logits: ', summarize=10000)
        # inputs = tf.Print(inputs, [inputs],
        #     message='inputs: ', summarize=10000)
        # targets = tf.Print(targets, [targets],
        #     message='targets: ', summarize=10000)

        # # prints for dynamic tensor shapes:
        # logits = tf.Print(logits, [tf.shape(logits)],
        #     message='logits shape: ', summarize=10000)
        # inputs = tf.Print(inputs, [tf.shape(inputs)],
        #     message='inputs shape: ', summarize=10000)
        # targets = tf.Print(targets, [tf.shape(targets)],
        #     message='targets shape: ', summarize=10000)

        loss_numerator, loss_denominator = common_layers.padded_cross_entropy(
            logits,
            targets,
            self._model_hparams.label_smoothing,
            weights_fn=self.targets_weights_fn
        )

        # additional losses based on inputs here
        # with tf.variable_scope("input-based-loss"):
        #     losses = tf.Variable(0.0, tf.float32)
        #     loss_numerator = tf.add(losses, loss_numerator)
        # end additional losses

        return loss_numerator, loss_denominator
