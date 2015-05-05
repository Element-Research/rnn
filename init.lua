require 'dpnn'

-- create global rnn table:
rnn = {}

torch.include('rnn', 'recursiveUtils.lua')

-- for testing:
torch.include('rnn', 'test.lua')

-- support modules
torch.include('rnn', 'ZeroGrad.lua')

-- recurrent modules
torch.include('rnn', 'AbstractRecurrent.lua')
torch.include('rnn', 'Recurrent.lua')
torch.include('rnn', 'LSTM.lua')
torch.include('rnn', 'Repeater.lua')
torch.include('rnn', 'Sequencer.lua')

-- recurrent criterions:
torch.include('rnn', 'RepeaterCriterion.lua')
torch.include('rnn', 'SequencerCriterion.lua')

-- prevent likely name conflicts
nn.rnn = rnn
