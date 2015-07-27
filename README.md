# rnn: recurrent neural networks #

This is a Recurrent Neural Network library that extends Torch's nn. 
You can use it to build RNNs, LSTMs, BRNNs, BLSTMs, and so forth and so on.
This library includes documentation for the following objects:

 * [AbstractRecurrent](#rnn.AbstractRecurrent) : an abstract class inherited by Recurrent and LSTM;
 * [Recurrent](#rnn.Recurrent) : a generalized recurrent neural network container;
 * [LSTM](#rnn.LSTM) : a vanilla Long-Short Term Memory module;
  * [FastLSTM](#rnn.FastLSTM) : a faster [LSTM](#rnn.LSTM);
 * [Sequencer](#rnn.Sequencer) : applies an encapsulated module to all elements in an input sequence;
 * [BiSequencer](#rnn.BiSequencer) : used for implementing Bidirectional RNNs and LSTMs;
 * [BiSequencerLM](#rnn.BiSequencerLM) : used for implementing Bidirectional RNNs and LSTMs for language models;
 * [Repeater](#rnn.Repeater) : repeatedly applies the same input to an AbstractRecurrent instance;
 * [SequencerCriterion](#rnn.SequencerCriterion) : sequentially applies the same criterion to a sequence of inputs and targets;
 * [RepeaterCriterion](#rnn.RepeaterCriterion) : repeatedly applies the same criterion with the same target on a sequence;


## Example ##
A very complete example with either the Penn Tree Bank or Google Billion Words datasets is available [here](https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua). 
 
<a name='rnn.AbstractRecurrent'></a>
## AbstractRecurrent ##
An abstract class inherited by [Recurrent](#rnn.Recurrent) and [LSTM](#rnn.LSTM).
The constructor takes a single argument :
```lua
rnn = nn.AbstractRecurrent(rho)
```
Argument `rho` is the maximum number of steps to backpropagate through time (BPTT).
Sub-classes can set this to a large number like 99999 if they want to backpropagate through 
the entire sequence whatever its length. Setting lower values of rho are 
useful when long sequences are forward propagated, but we only whish to 
backpropagate through the last `rho` steps, which means that the remainder 
of the sequence doesn't need to be stored (so no additional cost). 

### [recurrentModule] getStepModule(step) ###
Returns a module for time-step `step`. This is used internally by sub-classes 
to obtain copies of the internal `recurrentModule`. These copies share 
`parameters` and `gradParameters` but each have their own `output`, `gradInput` 
and any other intermediate states. 

### [output] updateOutput(input) ###
Forward propagates the input for the current step. The outputs or intermediate 
states of the previous steps are used recurrently. This is transparent to the 
caller as the previous outputs and intermediate states are memorized. This 
method also increments the `step` attribute by 1.

### updateGradInput(input, gradOutput) ###
It is important to understand that the actual BPTT happens in the `updateParameters`, 
`backwardThroughTime` or `backwardUpdateThroughTime` methods. So this 
method just keeps a copy of the `gradOutput`. These are stored in a 
table in the order that they were provided.

### accGradParameters(input, gradOutput, scale) ###
Again, the actual BPTT happens in the `updateParameters`, 
`backwardThroughTime` or `backwardUpdateThroughTime` methods.
So this method just keeps a copy of the `scales` for later.

### backwardThroughTime() ###
This method calls `updateGradInputThroughTime` followed by `accGradParametersThroughTime`.
This is where the actual BPTT happens.

### updateGradInputThroughTime() ###
Iteratively calls `updateGradInput` for all time-steps in reverse order 
(from the end to the start of the sequence). Returns the `gradInput` of 
the first time-step.

### accGradParametersThroughTime() ###
Iteratively calls `accGradParameters` for all time-steps in reverse order 
(from the end to the start of the sequence). 

### accUpdateGradParametersThroughTime(learningRate) ###
Iteratively calls `accUpdateGradParameters` for all time-steps in reverse order 
(from the end to the start of the sequence). 

### backwardUpdateThroughTime(learningRate) ###
This method calls `updateGradInputThroughTime` and 
`accUpdateGradParametersThroughTime(learningRate)` and returns the `gradInput` 
of the first step. 

### updateParameters(learningRate) ###
Unless `backwardThroughTime` or `accGradParameters` where called since
the last call to `updateOutput`, `backwardUpdateThroughTime` is called.
Otherwise, it calls `updateParameters` on all encapsulated Modules.

### recycle(offset) ###
This method goes hand in hand with `forget`. It is useful when the current
time-step is greater than `rho`, at which point it starts recycling 
the oldest `recurrentModule` `sharedClones`, 
such that they can be reused for storing the next step. This `offset` 
is used for modules like `nn.Recurrent` that use a different module 
for the first step. Default offset is 0.

<a name='rnn.AbstractRecurrent.forget'></a>
### forget(offset) ###
This method brings back all states to the start of the sequence buffers, 
i.e. it forgets the current sequence. It also resets the `step` attribute to 1.
It is highly recommended to call `forget` after each parameter update. 
Otherwise, the previous state will be used to activate the next, which 
will often lead to instability. This is caused by the previous state being
the result of now changed parameters. It is also good practice to call 
`forget` at the start of each new sequence.

### training() ###
In training mode, the network remembers all previous `rho` (number of time-steps)
states. This is necessary for BPTT. 

### evaluate() ###
During evaluation, since their is no need to perform BPTT at a later time, 
only the previous step is remembered. This is very efficient memory-wise, 
such that evaluation can be performed using potentially infinite-length 
sequence.
 
<a name='rnn.Recurrent'></a>
## Recurrent ##
References :
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

A [composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers) for implementing Recurrent Neural Networks (RNN), excluding the output layer. 

The `nn.Recurrent(start, input, feedback, [transfer, rho, merge])` constructor takes 5 arguments:
 * `start` : the size of the output (excluding the batch dimension), or a Module that will be inserted between the `input` Module and `transfer` module during the first step of the propagation. When `start` is a size (a number or `torch.LongTensor`), then this *start* Module will be initialized as `nn.Add(start)` (see Ref. A).
 * `input` : a Module that processes input Tensors (or Tables). Output must be of same size as `start` (or its output in the case of a `start` Module), and same size as the output of the `feedback` Module.
 * `feedback` : a Module that feedbacks the previous output Tensor (or Tables) up to the `transfer` Module.
 * `transfer` : a non-linear Module used to process the element-wise sum of the `input` and `feedback` module outputs, or in the case of the first step, the output of the *start* Module.
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Due to the vanishing gradients effect, references A and B recommend `rho = 5` (or lower). Defaults to 5.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input` and `feedback` Module before being forwarded through the `transfer` Module.
 
An RNN is used to process a sequence of inputs. 
Each step in the sequence should be propagated by its own `forward` (and `backward`), 
one `input` (and `gradOutput`) at a time. 
Each call to `forward` keeps a log of the intermediate states (the `input` and many `Module.outputs`) 
and increments the `step` attribute by 1. 
A call to `backward` doesn't result in a `gradInput`. It only keeps a log of the current `gradOutput` and `scale`.
Back-Propagation Through Time (BPTT) is done when the `updateParameters` or `backwardThroughTime` method
is called. The `step` attribute is only reset to 1 when a call to the `forget` method is made. 
In which case, the Module is ready to process the next sequence (or batch thereof).
Note that the longer the sequence, the more memory will be required to store all the 
`output` and `gradInput` states (one for each time step). 

To use this module with batches, we suggest using different 
sequences of the same size within a batch and calling `updateParameters` 
every `rho` steps and `forget` at the end of the Sequence. 

Note that calling the `evaluate` method turns off long-term memory; 
the RNN will only remember the previous output. This allows the RNN 
to handle long sequences without allocating any additional memory.

Example :
```lua
require 'rnn'

batchSize = 8
rho = 5
hiddenSize = 10
nIndex = 10000
-- RNN
r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, nIndex))
rnn:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- dummy dataset (task is to predict next item, given previous)
sequence = torch.randperm(nIndex)

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
updateInterval = 4
i = 1
while true do
   -- a batch of inputs
   local input = sequence:index(1, offsets)
   local output = rnn:forward(input)
   -- incement indices
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   local target = sequence:index(1, offsets)
   local err = criterion:forward(output, target)
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)
   
   i = i + 1
   -- note that updateInterval < rho
   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      r:updateParameters(lr)
   end
end
```

Note that `nn.Recurrent` can be decorated with a [nn.Sequencer](#rnn.Sequencer) such that an entire sequence (a table) can be presented with a single `forward/backward` call.

<a name='rnn.LSTM'></a>
## LSTM ##
References :
 * A. [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/pdf/1303.5778v1.pdf)
 * B. [Long-Short Term Memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
 * C. [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
 * D. [nngraph LSTM implementation on github](https://github.com/wojzaremba/lstm)

This is an implementation of a vanilla Long-Short Term Memory module. 
We used Ref. A's LSTM as a blueprint for this module as it was the most concise.
Yet it is also the vanilla LSTM described in Ref. C. 

The `nn.LSTM(inputSize, outputSize, [rho])` constructor takes 3 arguments:
 * `inputSize` : a number specifying the size of the input;
 * `outputSize` : a number specifying the size of the output;
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Defaults to 9999.

![LSTM](doc/image/LSTM.png) 

The actual implementation corresponds to the following algorithm:
```lua
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z(t)                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
h[t] = o[t]tanh(c[t])                                                (6)
```
where `W[s->q]` is the weight matrix from `s` to `q`, `t` indexes the time-step,
`b[1->q]` are the biases leading into `q`, `σ()` is `Sigmoid`, `x[t]` is the input,
`i[t]` is the input gate (eq. 1), `f[t]` is the forget gate (eq. 2), 
`z[t]` is the input to the cell (which we call the hidden) (eq. 3), 
`c[t]` is the cell (eq. 4), `o[t]` is the output gate (eq. 5), 
and `h[t]` is the output of this module (eq. 6). Also note that the 
weight matrices from cell to gate vectors are diagonal `W[c->s]`, where `s` 
is `i`,`f`, or `o`.

As you can see, unlike [Recurrent](#rnn.Recurrent), this 
implementation isn't generic enough that it can take arbitrary component Module
definitions at construction. However, the LSTM module can easily be adapted 
through inheritance by overriding the different factory methods :
  * `buildGate` : builds generic gate that is used to implement the input, forget and output gates;
  * `buildInputGate` : builds the input gate (eq. 1). Currently calls `buildGate`;
  * `buildForgetGate` : builds the forget gate (eq. 2). Currently calls `buildGate`;
  * `buildHidden` : builds the hidden (eq. 3);
  * `buildCell` : builds the cell (eq. 4);
  * `buildOutputGate` : builds the output gate (eq. 5). Currently calls `buildGate`;
  * `buildModel` : builds the actual LSTM model which is used internally (eq. 6).
  
<a name='rnn.FastLSTM'></a>
## FastLSTM ##

A faster version of the [LSTM](#rnn.LSTM). 
Basically, the input, forget and output gates, as well as the hidden state are computed at one fell swoop.
  
<a name='rnn.Sequencer'></a>
## Sequencer ##

The `nn.Sequencer(module)` constructor takes a single argument, `module`, which is the module 
to be applied from left to right, on each element of the input sequence.

```lua
seq = nn.Sequencer(module)
```

This Module is a kind of [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) 
used to abstract away the intricacies of AbstractRecurrent `modules`. While the latter 
require a sequence to be presented one input at a time, each with its own call to `forward` (and `backward`),
the Sequencer forwards an input sequence (a table) into an output sequence (a table of the same length).
It also takes care of calling `forget`, `backwardThroughTime` and other such AbstractRecurrent-specific methods.

For example, `rnn` : an instance of nn.AbstractRecurrent, can forward an `input` sequence one forward at a time:
```lua
input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
rnn:forward(input[1])
rnn:forward(input[2])
rnn:forward(input[3])
```

Equivalently, we can use a Sequencer to forward the entire `input` sequence at once:

```lua
seq = nn.Sequencer(rnn)
seq:forward(input)
```

The `Sequencer` can also take non-recurrent Modules (i.e. non-AbstractRecurrent instances) and apply it to each 
input to produce an output table of the same length. 
This is especially useful for processing variable length sequences (tables).

Note that for now, it is only possible to decorate either recurrent or non-recurrent Modules. 
Specifically, it cannot handle non-recurrent Modules containing recurrent Modules. 
Instead, either Modules should be encapsulated by its own `Sequencer`. This may change in the future.

### remember([mode]) ###
When `mode='both'` (the default), the Sequencer will not call [forget](#nn.AbstractRecurrent.forget) at the start of 
each call to `forward`, which is the default behavior of the class. 
This behavior is only applicable to decorated AbstractRecurrent `modules`.
Accepted values for argument `mode` are as follows :

 * 'eval' only affects evaluation (recommended for RNNs)
 * 'train' only affects training
 * 'neither' affects neither training nor evaluation (default behavior of the class)
 * 'both' affects both training and evaluation (recommended for LSTMs)

### forget() ###
Calls the decorated AbstractRecurrent module's `forget` method.

<a name='rnn.BiSequencer'></a>
## BiSequencer ##
Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing Bidirectional RNNs and LSTMs.

```lua
brnn = nn.BiSequencer(fwd, [bwd, merge])
```

The input to the module is a sequence (a table) of tensors
and the output is a sequence (a table) of tensors of the same length.
Applies a `fwd` rnn (an [AbstractRecurrent](#rnn.AbstractRecurrent) instance to each element in the sequence in
forward order and applies the `bwd` rnn in reverse order (from last element to first element).
The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

For each step (in the original sequence), the outputs of both rnns are merged together using
the `merge` module (defaults to `nn.JoinTable(1,1)`). 
If `merge` is a number, it specifies the [JoinTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable)
constructor's `nInputDim` argument. Such that the `merge` module is then initialized as :

```lua
merge = nn.JoinTable(1,merge)
```

Internally, the `BiSequencer` is implemented by decorating a structure of modules that makes 
use of 3 Sequencers for the forward, backward and merge modules.

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

<a name='rnn.BiSequencerLM'></a>
## BiSequencerLM ##

Applies encapsulated `fwd` and `bwd` rnns to an input sequence in forward and reverse order.
It is used for implementing Bidirectional RNNs and LSTMs for Language Models (LM).

```lua
brnn = nn.BiSequencerLM(fwd, [bwd, merge])
```

The input to the module is a sequence (a table) of tensors
and the output is a sequence (a table) of tensors of the same length.
Applies a `fwd` rnn (an [AbstractRecurrent](#rnn.AbstractRecurrent) instance to the 
first `N-1` elements in the sequence in forward order.
Applies the `bwd` rnn in reverse order to the last `N-1` elements (from second-to-last element to first element).
This is the main difference of this module with the [BiSequencer](#rnn.BiSequencer).
The latter cannot be used for language modeling because the `bwd` rnn would be trained to predict the input it had just be fed as input.

![BiDirectionalLM](doc/image/bidirectionallm.png)

The `bwd` rnn defaults to:

```lua
bwd = fwd:clone()
bwd:reset()
```

While the `fwd` rnn will output representations for the last `N-1` steps,
the `bwd` rnn will output representations for the first `N-1` steps.
The missing outputs for each rnn ( the first step for the `fwd`, the last step for the `bwd`)
will be filled with zero Tensors of the same size the commensure rnn's outputs.
This way they can be merged. If `nn.JoinTable` is used (the default), then the first 
and last output elements will be padded with zeros for the missing `fwd` and `bwd` rnn outputs, respectively.

For each step (in the original sequence), the outputs of both rnns are merged together using
the `merge` module (defaults to `nn.JoinTable(1,1)`). 
If `merge` is a number, it specifies the [JoinTable](https://github.com/torch/nn/blob/master/doc/table.md#nn.JoinTable)
constructor's `nInputDim` argument. Such that the `merge` module is then initialized as :

```lua
merge = nn.JoinTable(1,merge)
```

Similarly to a [Sequencer](#rnn.Sequencer), the sequences in a batch must have the same size.
But the sequence length of each batch can vary.

Note that LMs implemented with this module will not be classical LMs as they won't measure the 
probability of a word given the previous words. Instead, they measure the probabiliy of a word
given the surrounding words, i.e. context. While for mathematical reasons you may not be able to use this to measure the 
probability of a sequence of words (like a sentence), 
you can still measure the pseudo-likeliness of such a sequence (see [this](http://arxiv.org/pdf/1504.01575.pdf) for a discussion).

<a name='rnn.Repeater'></a>
## Repeater ##
This Module is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern) similar to [Sequencer].
It differs in that the sequence length is fixed before hand and the input is repeatedly forwarded 
through the wrapped `module` to produce an output table of length `nStep`:
```lua
r = nn.Repeater(module, nStep)
```
Argument `module` should be an `AbstractRecurrent` instance.
This is useful for implementing models like [RCNNs](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf),
which are repeatedly presented with the same input.

<a name='rnn.SequencerCriterion'></a>
## SequencerCriterion ##
This Criterion is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern):
```lua
c = nn.SequencerCriterion(criterion)
``` 
Both the `input` and `target` are expected to be a sequence (a table). 
For each step in the sequence, the corresponding elements of the input and target tables 
will be applied to the `criterion`.
The output of `forward` is the sum of all individual losses in the sequence.
This is useful when used in conjuction with a [Sequencer](#rnn.Sequencer).

WARNING : assumes that the decorated criterion is stateless, i.e. a `backward` shouldn't need to be preceded by a commensurate `forward`.

<a name='rnn.RepeaterCriterion'></a>
## RepeaterCriterion ##
This Criterion is a [decorator](http://en.wikipedia.org/wiki/Decorator_pattern):
```lua
c = nn.RepeaterCriterion(criterion)
``` 
The `input` is expected to be a sequence (a table). A single `target` is 
repeatedly applied using the same `criterion` to each element in the `input` sequence.
The output of `forward` is the sum of all individual losses in the sequence.
This is useful for implementing models like [RCNNs](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf),
which are repeatedly presented with the same target.
