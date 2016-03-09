require 'nn'

local basic_utils = {}

-------------------------------------------------------------------------------
-- Basic Module - nn.ScaleShift
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.ScaleShift', 'nn.Module')
function layer:__init(scale, shift)
  parent.__init(self)
  self.sc = scale
  self.sh = shift
end
function layer:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.output:mul(self.sc):add(self.sh)
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:mul(self.sc)
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Basic Module - nn.Scale
-------------------------------------------------------------------------------
-- layer that scale the input so that we can add weights on nn.Module.
-- it's like nn.Identity()*const, but I haven't found such scaling module in module.
local scale, parent = torch.class('nn.Scale', 'nn.Module')
function scale:__init(s)
  parent.__init(self)
  self.s = s
end
function scale:updateOutput(input)
  if self.s == 1 then self.output = input; return self.output end
  self.output:resizeAs(input):copy(input)
  self.output:mul(self.s)
  return self.output
end
function scale:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  if self.s ~= 0 then
    self.gradInput:mul(self.s)
  else
    self.gradInput:zero()
  end
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Basic Module - nn.FeatExpander
-------------------------------------------------------------------------------
-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:resetExpandSize(n)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- output size
  local out_dims = input:size()
  out_dims[1] = out_dims[1] * self.n
  self.output:resize(out_dims)
  -- expand size
  local expand_size = input:size()
  expand_size[1] = self.n
  -- simply expands out the features. Performs a copy information
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k} }]:expand(expand_size) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end


return basic_utils