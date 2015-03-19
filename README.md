# rnn: recurrent 'nn' components

This is a Recurrent Neural Network library that extends Torch's nn. 
You can use it to build RNNs, LSTMs, BRNNs, BLSTMs, and so forth and so on.

## Library Documentation ##
This section includes documentation for the following objects:
 * [Recurrent](#nnx.Recurrent) : a generalized recurrent neural network container;
 
<a name='rnn.Recurrent'/>
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

Note that this won't work with `input` and `feedback` modules that use more than their
`output` attribute to keep track of their internal state between 
calls to `forward` and `backward`.

