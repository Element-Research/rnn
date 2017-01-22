--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'rnn'

version = 1.3 -- Added multiple layers and merged with seqLSTM example

local opt = {}
opt.learningRate = 0.1
opt.hiddenSize = 6
opt.numLayers = 1
opt.useSeqLSTM = true -- faster implementation of LSTM + Sequencer
opt.vocabSize = 7
opt.seqLen = 7 -- length of the encoded sequence (with padding)
opt.niter = 1000

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function forwardConnect(enc, dec)
   for i=1,#enc.lstmLayers do
      if opt.useSeqLSTM then
         dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].output[opt.seqLen]
         dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cell[opt.seqLen]
      else
         dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[opt.seqLen])
         dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[opt.seqLen])
      end
   end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function backwardConnect(enc, dec)
   for i=1,#enc.lstmLayers do
      if opt.useSeqLSTM then
         enc.lstmLayers[i].userNextGradCell = dec.lstmLayers[i].userGradPrevCell
         enc.lstmLayers[i].gradPrevOutput = dec.lstmLayers[i].userGradPrevOutput
      else
         enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
         enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
      end
   end
end

-- Encoder
local enc = nn.Sequential()
enc:add(nn.LookupTableMaskZero(opt.vocabSize, opt.hiddenSize))
enc.lstmLayers = {}
for i=1,opt.numLayers do
   if opt.useSeqLSTM then
      enc.lstmLayers[i] = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
      enc.lstmLayers[i]:maskZero()
      enc:add(enc.lstmLayers[i])
   else
      enc.lstmLayers[i] = nn.LSTM(opt.hiddenSize, opt.hiddenSize):maskZero(1)
      enc:add(nn.Sequencer(enc.lstmLayers[i]))
   end
end
enc:add(nn.Select(1, -1))

-- Decoder
local dec = nn.Sequential()
dec:add(nn.LookupTableMaskZero(opt.vocabSize, opt.hiddenSize))
dec.lstmLayers = {}
for i=1,opt.numLayers do
   if opt.useSeqLSTM then
      dec.lstmLayers[i] = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
      dec.lstmLayers[i]:maskZero()
      dec:add(dec.lstmLayers[i])
   else
      dec.lstmLayers[i] = nn.LSTM(opt.hiddenSize, opt.hiddenSize):maskZero(1)
      dec:add(nn.Sequencer(dec.lstmLayers[i]))
   end
end
dec:add(nn.Sequencer(nn.MaskZero(nn.Linear(opt.hiddenSize, opt.vocabSize), 1)))
dec:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))

-- Some example data (batchsize = 2) with variable length input and output sequences

-- The input sentences to the encoder, padded with zeros from the left
local encInSeq = torch.Tensor({{0,0,0,0,1,2,3},{0,0,0,4,3,2,1}}):t()
-- The input sentences to the decoder, padded with zeros from the right.
-- Label '6' represents the start of a sentence (GO).
local decInSeq = torch.Tensor({{6,1,2,3,4,0,0,0},{6,5,4,3,2,1,0,0}}):t()

-- The expected output from the decoder (it will return one character per time-step),
-- padded with zeros from the right
-- Label '7' represents the end of sentence (EOS).
local decOutSeq = torch.Tensor({{1,2,3,4,7,0,0,0},{5,4,3,2,1,7,0,0}}):t()

for i=1,opt.niter do
   enc:zeroGradParameters()
   dec:zeroGradParameters()

   -- Forward pass
   local encOut = enc:forward(encInSeq)
   forwardConnect(enc, dec)
   local decOut = dec:forward(decInSeq)
   --print(decOut)
   local err = criterion:forward(decOut, decOutSeq)
   
   print(string.format("Iteration %d ; NLL err = %f ", i, err))

   -- Backward pass
   local gradOutput = criterion:backward(decOut, decOutSeq)
   dec:backward(decInSeq, gradOutput)
   backwardConnect(enc, dec)
   local zeroTensor = torch.Tensor(encOut):zero()
   enc:backward(encInSeq, zeroTensor)

   dec:updateParameters(opt.learningRate)
   enc:updateParameters(opt.learningRate)
end
