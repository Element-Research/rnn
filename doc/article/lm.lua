input = {}
for i=1,rho do
   table.insert(input, torch.Tensor(batchSize):random(1,nIndex))
end
output = rnn:forward(input)
assert(#output == #input)