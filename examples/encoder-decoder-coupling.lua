--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'nn'
require 'rnn'

torch.manualSeed(123)

version = 1.1 --supports both online and mini-batch training

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function forwardConnect(encLSTM, decLSTM)
  decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.inputSeqLen])
  decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.inputSeqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function backwardConnect(encLSTM, decLSTM)
  encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
  encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

function main()
  opt = {}
  opt.learningRate = 0.1
  opt.hiddenSz = 2
  opt.vocabSz = 5
  opt.inputSeqLen = 3 -- length of the encoded sequence

  -- Some example data
  local encInSeq, decInSeq, decOutSeq = torch.Tensor({{1,2,3},{3,2,1}}), torch.Tensor({{1,2,3,4},{4,3,2,1}}), torch.Tensor({{2,3,4,1},{1,2,4,3}})
  decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)
  
  -- Encoder
  local enc = nn.Sequential()
  enc:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
  enc:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
  local encLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
  enc:add(nn.Sequencer(encLSTM))
  enc:add(nn.SelectTable(-1))

  -- Decoder
  local dec = nn.Sequential()
  dec:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
  dec:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
  local decLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
  dec:add(nn.Sequencer(decLSTM))
  dec:add(nn.Sequencer(nn.Linear(opt.hiddenSz, opt.vocabSz)))
  dec:add(nn.Sequencer(nn.LogSoftMax()))

  local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

  local encParams, encGradParams = enc:getParameters()
  local decParams, decGradParams = dec:getParameters()

  enc:zeroGradParameters()
  dec:zeroGradParameters()

  -- Forward pass
  local encOut = enc:forward(encInSeq)
  forwardConnect(encLSTM, decLSTM)
  local decOut = dec:forward(decInSeq)
  local Edec = criterion:forward(decOut, decOutSeq)

  -- Backward pass
  local gEdec = criterion:backward(decOut, decOutSeq)
  dec:backward(decInSeq, gEdec)
  backwardConnect(encLSTM, decLSTM)
  local zeroTensor = torch.Tensor(2):zero()
  enc:backward(encInSeq, zeroTensor)

  --
  -- You would normally do something like this now:
  --   dec:updateParameters(opt.learningRate)
  --   enc:updateParameters(opt.learningRate)
  --
  -- Here, we do a numerical gradient check to make sure the coupling is correct:
  --
  local tester = torch.Tester()
  local tests = {}
  local eps = 1e-5

  function tests.gradientCheck()
    local decGP_est, encGP_est = torch.DoubleTensor(decGradParams:size()), torch.DoubleTensor(encGradParams:size())

    -- Easy function to do forward pass over coupled network and get error
    function forwardPass()
      local encOut = enc:forward(encInSeq)
      forwardConnect(encLSTM, decLSTM)
      local decOut = dec:forward(decInSeq)
      local E = criterion:forward(decOut, decOutSeq)
      return E
    end

    -- Check encoder
    for i = 1, encGradParams:size(1) do
      -- Forward with \theta+eps
      encParams[i] = encParams[i] + eps
      local C1 = forwardPass()
      -- Forward with \theta-eps
      encParams[i] = encParams[i] - 2 * eps
      local C2 = forwardPass()

      encParams[i] = encParams[i] + eps
      encGP_est[i] = (C1 - C2) / (2 * eps)
    end
    tester:assertTensorEq(encGradParams, encGP_est, eps, "Numerical gradient check for encoder failed")

    -- Check decoder
    for i = 1, decGradParams:size(1) do
      -- Forward with \theta+eps
      decParams[i] = decParams[i] + eps
      local C1 = forwardPass()
      -- Forward with \theta-eps
      decParams[i] = decParams[i] - 2 * eps
      local C2 = forwardPass()

      decParams[i] = decParams[i] + eps
      decGP_est[i] = (C1 - C2) / (2 * eps)
    end
    tester:assertTensorEq(decGradParams, decGP_est, eps, "Numerical gradient check for decoder failed")
  end

  tester:add(tests)
  tester:run()
end

main()
