# rnn: recurrent 'nn' components

This is a Recurrent Neural Network library that extends Torch's nn. 
You can use it to build RNNs, LSTMs, BRNNs, BLSTMs, and so forth and so on.

## Library Documentation ##
This section includes documentation for the following objects:
 * [AbstractRecurrent](#rnn.AbstractRecurrent) : an abstract class inherited by Recurrent and LSTM;
 * [Recurrent](#rnn.Recurrent) : a generalized recurrent neural network container;
 * [LSTM](#rnn.LSTM) : a vanilla Long-Short Term Memory module;
 * [Sequencer](#rnn.Sequencer) : applies an encapsulated module to all elements in an input sequence;
 * [Repeater](#rnn.Repeater) : repeatedly applies the same input to an AbstractRecurrent instance;
 * [RepeaterCriterion](#rnn.RepeaterCriterion) : repeatedly applies the same criterion with the same target on a sequence;
 
<a name='rnn.AbstractRecurrent'></a>
### AbstractRecurrent ###
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
### Recurrent ###
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
require 'nnx'

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

