------------------------------------------------------------------------
--[[ SeqBRNN ]] --
-- Bi-directional RNN using two SeqLSTM modules.
-- Input is a tensor e.g time x batch x inputdim.
-- Output is a tensor of the same length e.g time x batch x outputdim.
-- Applies a forward rnn to input tensor in forward order
-- and applies a backward rnn in reverse order.
-- Reversal of the sequence happens on the time dimension.
-- For each step, the outputs of both rnn are merged together using
-- the merge module (defaults to nn.CAddTable() which sums the activations).
------------------------------------------------------------------------
local SeqBRNN, parent = torch.class('nn.SeqBRNN', 'nn.Container')

function SeqBRNN:__init(inputDim, hiddenDim, batchFirst, merge)
    self.forwardModule = nn.SeqLSTM(inputDim, hiddenDim)
    self.backwardModule = nn.SeqLSTM(inputDim, hiddenDim)
    self.merge = merge
    if not self.merge then
        self.merge = nn.CAddTable()
    end
    self.dim = 1
    local backward = nn.Sequential()
    backward:add(nn.SeqReverseSequence(self.dim)) -- reverse
    backward:add(self.backwardModule)
    backward:add(nn.SeqReverseSequence(self.dim)) -- unreverse

    local concat = nn.ConcatTable()
    concat:add(self.forwardModule):add(backward)

    local brnn = nn.Sequential()
    brnn:add(concat)
    brnn:add(self.merge)
    if(batchFirst) then
        -- Insert transposes before and after the brnn.
        brnn:insert(nn.Transpose({1, 2}), 1)
        brnn:insert(nn.Transpose({1, 2}))
    end

    parent.__init(self)

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.module = brnn
    -- so that it can be handled like a Container
    self.modules[1] = brnn
end

function SeqBRNN:updateOutput(input)
    self.output = self.module:updateOutput(input)
    return self.output
end

function SeqBRNN:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end

function SeqBRNN:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(input, gradOutput, scale)
end

function SeqBRNN:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function SeqBRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function SeqBRNN:__tostring__()
    if self.module.__tostring__ then
        return torch.type(self) .. ' @ ' .. self.module:__tostring__()
    else
        return torch.type(self) .. ' @ ' .. torch.type(self.module)
    end
end