require 'dpnn'
dpnn.version = dpnn.version or 0
assert(dpnn.version > 1, "Please update dpnn : luarocks install dpnn")

-- create global rnn table:
rnn = {}
rnn.version = 2

unpack = unpack or table.unpack

torch.include('rnn', 'recursiveUtils.lua')

-- extensions to nn.Module
torch.include('rnn', 'Module.lua')

-- for testing:
torch.include('rnn', 'test.lua')

-- support modules
torch.include('rnn', 'ZeroGrad.lua')
torch.include('rnn', 'LinearNoBias.lua')

-- recurrent modules
torch.include('rnn', 'LookupTableMaskZero.lua')
torch.include('rnn', 'MaskZero.lua')
torch.include('rnn', 'AbstractRecurrent.lua')
torch.include('rnn', 'Recurrent.lua')
torch.include('rnn', 'LSTM.lua')
torch.include('rnn', 'FastLSTM.lua')
torch.include('rnn', 'GRU.lua')
torch.include('rnn', 'SAdd.lua')
torch.include('rnn', 'Recursor.lua')
torch.include('rnn', 'Recurrence.lua')

torch.include('rnn', 'AbstractSequencer.lua')
torch.include('rnn', 'Repeater.lua')
torch.include('rnn', 'Sequencer.lua')
torch.include('rnn', 'BiSequencer.lua')
torch.include('rnn', 'BiSequencerLM.lua')
torch.include('rnn', 'RecurrentAttention.lua')

-- recurrent criterions:
torch.include('rnn', 'RepeaterCriterion.lua')
torch.include('rnn', 'SequencerCriterion.lua')
torch.include('rnn', 'MaskZeroCriterion.lua')

-- prevent likely name conflicts
nn.rnn = rnn
