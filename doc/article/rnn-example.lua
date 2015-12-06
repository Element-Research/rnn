-- generate some dummy inputs and gradOutputs sequences
inputs, gradOutputs = {}, {}
for step=1,rho do
   inputs[step] = torch.randn(batchSize,inputSize)
   gradOutputs[step] = torch.randn(batchSize,inputSize)
end

-- an AbstractRecurrent instance
rnn = nn.Recurrent(
   hiddenSize, -- size of the input layer
   nn.Linear(inputSize,outputSize), -- input layer
   nn.Linear(outputSize, outputSize), -- recurrent layer
   nn.Sigmoid(), -- transfer function
   rho -- maximum number of time-steps for BPTT
)

-- feed-forward and backpropagate through time like this :
for step=1,rho do
   rnn:forward(inputs[step])
   rnn:backward(inputs[step], gradOutputs[step])
end
rnn:backwardThroughTime() -- call backward on the internal modules
gradInputs = rnn.gradInputs
rnn:updateParameters(0.1)
rnn:forget() -- resets the time-step counter