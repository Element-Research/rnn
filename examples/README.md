# Examples

This directory contains various training scripts. 

Torch blog posts
 * The torch.ch blog contains detailed posts about the *rnn* package.
 1. [recurrent-visual-attention.lua](recurrent-visual-attention.lua): training script used in [Recurrent Model for Visual Attention](http://torch.ch/blog/2015/09/21/rmva.html). Implements the REINFORCE learning rule to learn an attention mechanism for classifying MNIST digits, sometimes translated.
 2. [noise-contrastive-esimate.lua](noise-contrastive-estimate.lua): one of two training scripts used in [Language modeling a billion words](http://torch.ch/blog/2016/07/25/nce.html). Single-GPU script for training recurrent language models on the Google billion words dataset.
 3. [multigpu-nce-rnnlm.lua](multigpu-nce-rnnlm.lua) : 4-GPU version of `noise-contrastive-estimate.lua` for training larger multi-GPU models. Two of two training scripts used in the [Language modeling a billion words](http://torch.ch/blog/2016/07/25/nce.html).

Simple training scripts. 
 * Showcases the fundamental principles of the package. In chronological order of introduction date.
 1. [simple-recurrent-network.lua](simple-recurrent-network.lua): uses the `nn.Recurrent` module to instantiate a Simple RNN. Illustrates the first AbstractRecurrent instance in action. It has since been surpassed by the more flexible `nn.Recursor` and `nn.Recurrence`. The `nn.Recursor` class decorates any module to make it conform to the nn.AbstractRecurrent interface. The `nn.Recurrence` implements the recursive `h[t] <- forward(h[t-1], x[t])`. Together, `nn.Recursor` and `nn.Recurrence` can be used to implement a wide range of experimental recurrent architectures.
 2. [simple-sequencer-network.lua](simple-sequencer-network.lua): uses the `nn.Sequencer` module to accept a batch of sequences as `input` of size `seqlen x batchsize x ...`. Both tables and tensors are accepted as input and produce the same type of output (table->table, tensor->tensor). The `Sequencer` class abstract away the implementation of back-propagation through time. It also provides a `remember(['neither','both'])` method for triggering what the `Sequencer` remembers between iterations (forward,backward,update).
 3. [simple-recurrence-network.lua](simple-recurrence-network.lua): uses the `nn.Recurrence` module to define the h[t] <- sigmoid(h[t-1], x[t]) Simple RNN. Decorates it using `nn.Sequencer` so that an entire batch of sequences (`input`) can forward and backward propagated per update.
