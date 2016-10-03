local SeqDropout, Parent = torch.class('nn.SeqDropout', 'nn.Module')

function SeqDropout:__init(p,v1,inplace,stochasticInference)
	Parent.__init(self)
	self.p = p or 0.5
	self.train = true
	self.inplace = inplace
	self.stochastic_inference = stochasticInference or false
	-- version 2 scales output during training instead of evaluation
	self.v2 = not v1
	if self.p >= 1 or self.p < 0 then
		error('<SeqDropout> illegal percentage, must be 0 <= p < 1')
	end
	self.noise = torch.Tensor()
end

function SeqDropout:updateOutput(input)
	if self.inplace then
		self.output:set(input)
	else
		self.output:resizeAs(input):copy(input)
	end
	if self.p > 0 then
		if self.train or self.stochastic_inference then
			self.noise:resizeAs(self.output:narrow(1,1,1))
			self.noise:bernoulli(1-self.p)
			if self.v2 then
				self.noise:div(1-self.p)
			end
			self.noise=self.noise:expandAs(self.output)
			self.output:cmul(self.noise)
		elseif not self.v2 then
			self.output:mul(1-self.p)
		end
	end
	return self.output
end

function SeqDropout:updateGradInput(input, gradOutput)
	if self.inplace then
		self.gradInput:set(gradOutput)
	else
		self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	end
	if self.train then
		if self.p > 0 then
			self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
		end
	else
		if not self.v2 and self.p > 0 then
			self.gradInput:mul(1-self.p)
		end
	end
	return self.gradInput
end

function SeqDropout:setp(p)
	self.p = p
end

function SeqDropout:__tostring__()
	return string.format('%s(%f)', torch.type(self), self.p)
end


function SeqDropout:clearState()
	if self.noise then
		self.noise:set()
	end
	return Parent.clearState(self)
end
