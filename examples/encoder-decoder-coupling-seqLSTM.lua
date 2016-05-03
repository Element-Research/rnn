--[[

Example of "coupled" separate encoder and decoder networks using SeqLSTM, e.g. for sequence-to-sequence networks.

]]--

require 'rnn'

version = 1.0

local opt = {}
opt.learningRate = 0.1
opt.hiddenSize = 6
opt.vocabSize = 6
opt.seqLen = 3 -- length of the encoded sequence
opt.niter = 1000

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
local function forwardConnect(encLSTM, decLSTM)
   decLSTM.userPrevOutput = encLSTM.output[opt.seqLen]
   decLSTM.userPrevCell = encLSTM.cell[opt.seqLen]
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
local function backwardConnect(encLSTM, decLSTM)
   encLSTM.userNextGradCell = decLSTM.userGradPrevCell
   encLSTM.gradPrevOutput = decLSTM.userGradPrevOutput
end

-- Encoder
local enc = nn.Sequential()
enc:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
local encLSTM = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
enc:add(encLSTM)

-- Decoder
local dec = nn.Sequential()
dec:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
local decLSTM = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
dec:add(decLSTM)
dec:add(nn.SplitTable(1, 3))
dec:add(nn.Sequencer(nn.Linear(opt.hiddenSize, opt.vocabSize)))
dec:add(nn.Sequencer(nn.LogSoftMax()))

local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- Some example data (batchsize = 2)
-- The input sentences to the encoder. 
local encInSeq = torch.Tensor({{1,2,3},{3,2,1}}):t(1,2)
-- The input sentences to the decoder. Label '5' represents the start of a sentence (GO).
local decInSeq = torch.Tensor({{5,1,2,3,4},{5,4,3,2,1}}):t(1,2)
-- The expected output from the decoder (it will return one character per time-step).
-- Label '6' represents the end of sentence (EOS).
local decOutSeq = torch.Tensor({{1,2,3,4,6},{1,2,4,3,6}})
-- The decoder predicts one per timestep, so we split accordingly.
decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

for i=1,opt.niter do
   enc:zeroGradParameters()
   dec:zeroGradParameters()

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(encLSTM, decLSTM)
   local decOut = dec:forward(decInSeq)
   local err = criterion:forward(decOut, decOutSeq)
   
   print(string.format("Iteration %d ; NLL err = %f ", i, err))

   -- Backward pass
   local gradOutput = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gradOutput)
   backwardConnect(encLSTM, decLSTM)
   local zeroTensor = torch.Tensor(encOut):zero()
   enc:backward(encInSeq, zeroTensor)

   dec:updateParameters(opt.learningRate)
   enc:updateParameters(opt.learningRate)
end
