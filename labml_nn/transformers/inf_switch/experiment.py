"""
---
title: Switch Transformer Experiment
summary: This experiment trains a small switch transformer on tiny Shakespeare dataset.
---

# Switch Transformer Experiment

This is an annotated PyTorch experiment to train a switch transformer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/353770ce177c11ecaa5fb74452424f46)
"""

from typing import List

import torch
from labml import experiment, tracker, logger
from labml.configs import option
from labml.logger import Text
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex
from torch import nn

from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import Generator
from labml_nn.transformers.inf_switch import InfSwitchTransformer
from labml_nn.transformers.mlm import MLM


class TransformerMLM(nn.Module):
    """
    # Transformer based model for MLM
    """

    def __init__(self, *, encoder: InfSwitchTransformer, src_embed: Module, generator: Generator):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.generator = generator
        self.src_embed = src_embed
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x, = self.encoder(x, None)
        # Logits for the output
        y = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return y, None


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        # Transformer
        self.transformer = transformer
        # Final layer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Initialize the subsequent mask
        if self.mask is None or self.mask.size(0) != len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Token embeddings
        x = self.src_embed(x)
        # Run it through the transformer
        res, counts, route_prob, n_dropped, route_prob_max = self.transformer(x, self.mask)
        # Generate logits of the next token
        res = self.generator(res)
        #
        return res, counts, route_prob, n_dropped, route_prob_max




class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html)
    because it has the data pipeline implementations that we reuse here.
    We have implemented a custom training step form MLM.
    """

    model: AutoregressiveModel
    transformer: Module

    # Number of tokens
    n_tokens: int = 'n_tokens_mlm'
    # Tokens that shouldn't be masked
    no_mask_tokens: List[int] = []
    # Probability of masking a token
    masking_prob: float = 0.15
    # Probability of replacing the mask with a random token
    randomize_prob: float = 0.1
    # Probability of replacing the mask with original token
    no_change_prob: float = 0.1
    # [Masked Language Model (MLM) class](index.html) to generate the mask
    mlm: MLM

    # `[MASK]` token
    mask_token: int
    # `[PADDING]` token
    padding_token: int

    # Prompt to sample
    prompt: str = [
        "We are accounted poor citizens, the patricians good.",
        "What authority surfeits on would relieve us: if they",
        "would yield us but the superfluity, while it were",
        "wholesome, we might guess they relieved us humanely;",
        "but they think we are too dear: the leanness that",
        "afflicts us, the object of our misery, is as an",
        "inventory to particularise their abundance; our",
        "sufferance is a gain to them Let us revenge this with",
        "our pikes, ere we become rakes: for the gods know I",
        "speak this in hunger for bread, not in thirst for revenge.",
    ]


    # Token embedding size
    d_model: int = 128
    # Number of attention heads
    heads: int = 4
    # Dropout probability
    dropout: float = 0.0
    # Number of features in FFN hidden layer
    d_ff: int = 256
    # Number of transformer layers
    n_layers: int = 6
    # Number of experts
    n_experts: int = 4
    # Load balancing coefficient
    load_balancing_loss_ceof = 0.01
    # Whether to scale the chosen expert outputs by the routing probability
    is_scale_prob: bool = True
    # Whether to drop tokens
    drop_tokens: bool = False
    # Capacity factor to determine capacity of each model
    capacity_factor: float = 1.0


    def init(self):
        """
        ### Initialization
        """
        # Initialize tracking indicators
        tracker.set_scalar("lb_loss.*", False)
        tracker.set_scalar("route.*", False)
        tracker.set_scalar("dropped.*", False)
        # `[MASK]` token
        self.mask_token = self.n_tokens - 1
        # `[PAD]` token
        self.padding_token = self.n_tokens - 2

        # [Masked Language Model (MLM) class](index.html) to generate the mask
        self.mlm = MLM(padding_token=self.padding_token,
                       mask_token=self.mask_token,
                       no_mask_tokens=self.no_mask_tokens,
                       n_tokens=self.n_tokens,
                       masking_prob=self.masking_prob,
                       randomize_prob=self.randomize_prob,
                       no_change_prob=self.no_change_prob)

        # Accuracy metric (ignore the labels equal to `[PAD]`)
        self.accuracy = Accuracy(ignore_index=self.padding_token)
        # Cross entropy loss (ignore the labels equal to `[PAD]`)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.padding_token)
        #
        super().init()

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Move the input to the device
        data = batch[0].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Get the masked input and labels
        with torch.no_grad():
            data, labels = self.mlm(data)

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet.
            output, *_ = self.model(data)

        # Calculate and log the loss
        loss = self.loss_func(output.view(-1, output.shape[-1]), labels.view(-1))
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, labels)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()

    @torch.no_grad()
    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        # Empty tensor for data filled with `[PAD]`.
        data = torch.full((self.seq_len, len(self.prompt)), self.padding_token, dtype=torch.long)
        # Add the prompts one by one
        for i, p in enumerate(self.prompt):
            # Get token indexes
            d = self.text.text_to_i(p)
            # Add to the tensor
            s = min(self.seq_len, len(d))
            data[:s, i] = d[:s]
        # Move the tensor to current device
        data = data.to(self.device)

        # Get masked input and labels
        data, labels = self.mlm(data)
        # Get model outputs
        output, *_ = self.model(data)

        # Print the samples generated
        for j in range(data.shape[1]):
            # Collect output from printing
            log = []
            # For each token
            for i in range(len(data)):
                # If the label is not `[PAD]`
                if labels[i, j] != self.padding_token:
                    # Get the prediction
                    t = output[i, j].argmax().item()
                    # If it's a printable character
                    if t < len(self.text.itos):
                        # Correct prediction
                        if t == labels[i, j]:
                            log.append((self.text.itos[t], Text.value))
                        # Incorrect prediction
                        else:
                            log.append((self.text.itos[t], Text.danger))
                    # If it's not a printable character
                    else:
                        log.append(('*', Text.danger))
                # If the label is `[PAD]` (unmasked) print the original.
                elif data[i, j] < len(self.text.itos):
                    log.append((self.text.itos[data[i, j]], Text.subtle))

            # Print
            logger.log(log)


@option(Configs.n_tokens)
def n_tokens_mlm(c: Configs):
    """
    Number of tokens including `[PAD]` and `[MASK]`
    """
    return c.text.n_tokens + 2
# class Configs(NLPAutoRegressionConfigs):
#     """
#     ## Configurations
#
#     This extends [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html).
#
#     The default configs can and will be over-ridden when we start the experiment
#     """
#
#     model: TransformerMLM
#     transformer: Module
#
#     # Token embedding size
#     d_model: int = 128
#     # Number of attention heads
#     heads: int = 4
#     # Dropout probability
#     dropout: float = 0.0
#     # Number of features in FFN hidden layer
#     d_ff: int = 256
#     # Number of transformer layers
#     n_layers: int = 6
#     # Number of experts
#     n_experts: int = 4
#     # Load balancing coefficient
#     load_balancing_loss_ceof = 0.01
#     # Whether to scale the chosen expert outputs by the routing probability
#     is_scale_prob: bool = True
#     # Whether to drop tokens
#     drop_tokens: bool = False
#     # Capacity factor to determine capacity of each model
#     capacity_factor: float = 1.0
#
#     def init(self):
#         super().init()
#         # Initialize tracking indicators
#         tracker.set_scalar("lb_loss.*", False)
#         tracker.set_scalar("route.*", False)
#         tracker.set_scalar("dropped.*", False)
#
#     def step(self, batch: any, batch_idx: BatchIndex):
#         """
#         ### Training or validation step
#         """
#
#         # Move data to the device
#         data, target = batch[0].to(self.device), batch[1].to(self.device)
#
#         # Update global step (number of tokens processed) when in training mode
#         if self.mode.is_train:
#             tracker.add_global_step(data.shape[0] * data.shape[1])
#
#         # Whether to capture model outputs
#         with self.mode.update(is_log_activations=batch_idx.is_last):
#             # Get model outputs.
#             output, counts, route_prob, n_dropped, route_prob_max = self.model(data)
#
#         # Calculate and cross entropy loss
#         cross_entropy_loss = self.loss_func(output, target)
#         # Total number of tokens processed, $T$, in the current batch $\mathscr{B}$
#         total = counts.sum(dim=-1, keepdims=True)
#         # Fraction of tokens routed to each expert
#         # $$f_i = \frac{1}{T} \sum_{x \in \mathscr{B}} \mathbf{1} \{ \mathop{argmax} p(x), i \}$$
#         # $f_i$ is the count of tokens where the argmax of $p(x)$ is equal to $i$.
#         route_frac = counts / total
#         # Mean routing probability
#         # $$P_i = \frac{1}{T} \sum_{x \in \mathscr{B}} p_i (x)$$
#         route_prob = route_prob / total
#         # Load balancing loss
#         # $$\mathscr{L} = N \sum_{i=1}^N f_i \cdot P_i$$
#         # $\mathscr{L}$ is the loss for a single layer and here we are
#         # taking the sum of losses across all layers.
#         load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()
#
#         # Track stats
#         tracker.add('dropped.', total.new_tensor(n_dropped) / total)
#         tracker.add('route.min.', route_frac.min())
#         tracker.add('route.max.', route_frac.max())
#         tracker.add('route.std.', route_frac.std())
#         tracker.add('route.max_prob.', route_prob_max)
#         tracker.add("loss.", cross_entropy_loss)
#         tracker.add("lb_loss.", load_balancing_loss)
#
#         # Combined loss.
#         # The load balancing loss is multiplied by a coefficient $\alpha$ which is
#         # set to something small like $\alpha = 0.01$.
#         loss = cross_entropy_loss + self.load_balancing_loss_ceof * load_balancing_loss
#
#         # Calculate and log accuracy
#         self.accuracy(output, target)
#         self.accuracy.track()
#
#         # Train the model
#         if self.mode.is_train:
#             # Calculate gradients
#             loss.backward()
#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
#             # Take optimizer step
#             self.optimizer.step()
#             # Log the model parameters and gradients on last batch of every epoch
#             if batch_idx.is_last:
#                 tracker.add('model', self.model)
#             # Clear the gradients
#             self.optimizer.zero_grad()
#
#         # Save the tracked metrics
#         tracker.save()


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    # m = TransformerMLM(c.n_tokens, c.d_model, c.transformer)
    m = AutoregressiveModel(c.n_tokens, c.d_model, c.transformer)
    return m.to(c.device)


# @option(Configs.transformer)
# def switch_transformer(c: Configs):
#     """
#     ### Initialize the switch transformer
#     """
#     from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
#     from labml_nn.transformers import MultiHeadAttention
#     from labml_nn.transformers.feed_forward import FeedForward
#
#     return SwitchTransformer(
#         SwitchTransformerLayer(d_model=c.d_model,
#                                attn=MultiHeadAttention(c.heads, c.d_model, c.dropout),
#                                feed_forward=SwitchFeedForward(capacity_factor=c.capacity_factor,
#                                                               drop_tokens=c.drop_tokens,
#                                                               is_scale_prob=c.is_scale_prob,
#                                                               n_experts=c.n_experts,
#                                                               expert=FeedForward(c.d_model, c.d_ff, c.dropout),
#                                                               d_model=c.d_model),
#                                dropout_prob=c.dropout),
#         c.n_layers)
@option(Configs.transformer)
def inf_switch_transformer(c: Configs):
    """
    ### Initialize the switch transformer
    """
    from labml_nn.transformers.inf_switch import InfSwitchTransformer, InfSwitchTransformerLayer, InfSwitchFeedForward
    from labml_nn.transformers import MultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward

    return InfSwitchTransformer(
        InfSwitchTransformerLayer(d_model=c.d_model,
                                  attn=MultiHeadAttention(c.heads, c.d_model, c.dropout),
                                  feed_forward=InfSwitchFeedForward(capacity_factor=c.capacity_factor,
                                                                    drop_tokens=c.drop_tokens,
                                                                    is_scale_prob=c.is_scale_prob,
                                                                    n_experts=c.n_experts,
                                                                    expert=FeedForward(c.d_model, c.d_ff, c.dropout),
                                                                    d_model=c.d_model),
                                  dropout_prob=c.dropout),
        c.n_layers)


def main():
    """
    ### Run the experiment
    """
    # Create experiment
    experiment.create(name="mlm", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {
                        #    tokenizer': 'character',
                        # 'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.,
                        'optimizer.optimizer': 'Noam',
                        # 'prompt': 'It is',
                        # 'prompt_separator': '',

                        'transformer': 'inf_switch_transformer',
                        'n_experts': 25,

                        'drop_tokens': False,
                        'capacity_factor': 1.2,

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 32,
                        'epochs': 128,
                        'batch_size': 64,
                        'inner_iterations': 25,
                        })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


#
if __name__ == '__main__':
    main()
