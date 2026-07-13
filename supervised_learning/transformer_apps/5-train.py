#!/usr/bin/env python3
"""Module that creates and trains a transformer for translation."""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup for the transformer."""

    def __init__(self, dm, warmup_steps=4000):
        """Initializes the schedule.

        Args:
            dm: dimensionality of the model.
            warmup_steps: number of warmup steps.
        """
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Computes the learning rate for a given step.

        Args:
            step: the current training step.

        Returns:
            The learning rate.
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Creates and trains a transformer for Portuguese to English.

    Args:
        N: number of blocks in the encoder and decoder.
        dm: dimensionality of the model.
        h: number of heads.
        hidden: number of hidden units in the fully connected layers.
        max_len: maximum number of tokens per sequence.
        batch_size: batch size for training.
        epochs: number of epochs to train for.

    Returns:
        The trained model.
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden, input_vocab, target_vocab, max_len, max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(real, pred):
        """Computes the loss ignoring padded tokens."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """Computes the accuracy ignoring padded tokens."""
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inputs, target)) in enumerate(data.data_train):
            target_input = target[:, :-1]
            target_real = target[:, 1:]

            enc_mask, comb_mask, dec_mask = create_masks(
                inputs, target_input
            )

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inputs, target_input, True,
                    enc_mask, comb_mask, dec_mask
                )
                loss = loss_function(target_real, predictions)

            gradients = tape.gradient(
                loss, transformer.trainable_variables
            )
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            train_loss(loss)
            train_accuracy(accuracy_function(target_real, predictions))

            if batch % 50 == 0:
                print("Epoch {}, batch {}: loss {} accuracy {}".format(
                    epoch + 1, batch,
                    train_loss.result(), train_accuracy.result()
                ))

        print("Epoch {}: loss {} accuracy {}".format(
            epoch + 1, train_loss.result(), train_accuracy.result()
        ))

    return transformer
