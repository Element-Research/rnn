-- recurrent module
rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize))
      :add(nn.Linear(hiddenSize, hiddenSize)))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())
-- full RNN
rnn = nn.Sequential()
   :add(nn.Sequencer(nn.Recurrence(rm, hiddenSize, 1)))
   :add(nn.SelectTable(-1)) --select last element
   :add(nn.Linear(hiddenSize, nSentiment))
   :add(nn.LogSoftMax())
)