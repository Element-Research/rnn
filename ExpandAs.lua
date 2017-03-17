local ExpandAs, parent = torch.class('nn.ExpandAs', 'nn.Module')
-- expands the second input to match the first

function ExpandAs:__init()
  parent.__init(self)
  self.output = {}
  self.gradInput = {}

  self.sum1 = torch.Tensor()
  self.sum2 = torch.Tensor()
end

function ExpandAs:updateOutput(input)
  self.output[1] = input[1]
  self.output[2] = input[2]:expandAs(input[1])
  return self.output
end

function ExpandAs:updateGradInput(input, gradOutput)
  local b, db = input[2], gradOutput[2]
  local s1, s2 = self.sum1, self.sum2
  local sumSrc, sumDst = db, s1

  for i=1,b:dim() do
    if b:size(i) ~= db:size(i) then
      sumDst:sum(sumSrc, i)
      sumSrc = sumSrc == s1 and s2 or s1
      sumDst = sumDst == s1 and s2 or s1
    end
  end

  self.gradInput[1] = gradOutput[1]
  self.gradInput[2] = sumSrc

  return self.gradInput
end
