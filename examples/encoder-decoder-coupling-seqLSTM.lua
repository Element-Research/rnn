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

-- Encoder
local enc = nn.Sequential()
enc:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
local encLSTM = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
enc:add(encLSTM)

-- Decoder
local dec = nn.Sequential()
local dec_lookup = nn.ParallelTable()
dec_lookup:add(nn.Identity()) -- To pass along c0
dec_lookup:add(nn.Identity()) -- To pass along h0
dec_lookup:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
local decLSTM = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
dec:add(dec_lookup)
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
   local decInput = {encLSTM.cell:select(1, encLSTM.cell:size(1)), encOut:select(1, encOut:size(1)), decInSeq}
   local decOut = dec:forward(decInput)
   local err = criterion:forward(decOut, decOutSeq)
   
   print(string.format("Iteration %d ; NLL err = %f ", i, err))
   
   -- Backward pass
   local gradOutput = criterion:backward(decOut, decOutSeq)
   local dc0, dh0, _ = unpack(dec:backward(decInput, gradOutput))
   local zeroTensor = torch.Tensor(encOut):zero()
   -- TODO: dc0 where to input this?
   zeroTensor[zeroTensor:size(1)] = dh0 -- Copy gradient to encoder
   enc:backward(encInSeq, zeroTensor)

   dec:updateParameters(opt.learningRate)
   enc:updateParameters(opt.learningRate)
end
