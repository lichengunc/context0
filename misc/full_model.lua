require 'misc.basic_modules'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
local utils = require 'misc.utils'

local layer, parent = torch.class('nn.FullModel', 'nn.Module')

function layer:__init(cnn, opt)
	parent.__init(self)
	-- jemb options
	local visual_encoding_size = utils.getopt(opt, 'visual_encoding_size', 512)
	local visual_dropout = utils.getopt(opt, 'visual_dropout', 0)
	local init_norm = utils.getopt(opt, 'init_norm', 10)
	self.use_context = utils.getopt(opt, 'use_context', 1)  -- 0.none, 1.img, 2.window2, 3.window3
	self.use_dif = utils.getopt(opt, 'use_dif', 1) 					-- 0.none, 1.mean, 2.max, 3.weighted

	-- lm options
	local rnn_type = utils.getopt(opt, 'rnn_type', 'lstm')
	local rnn_size = utils.getopt(opt, 'rnn_size', 512)
	local vocab_size = utils.getopt(opt, 'vocab_size')
	local seq_length = utils.getopt(opt, 'seq_length')
	local word_encoding_size = utils.getopt(opt, 'word_encoding_size', 512)
	local word_dropout = utils.getopt(opt, 'word_dropout', 0.5)
	local seq_per_ref = utils.getopt(opt, 'seq_per_ref', 3)

	-- construct jemb
	local jembOpt = {}
	jembOpt.dropout = visual_dropout
	jembOpt.use_context = opt.use_context
	jembOpt.use_dif = opt.use_dif
	jembOpt.visual_encoding_size = visual_encoding_size
	jembOpt.init_norm = init_norm
	self.jemb = net_utils.build_jemb(cnn, cnn, jembOpt)
	-- construct word_embedding layer
	self.we = nn.LookupTable(vocab_size+1, word_encoding_size)
	-- construct language model
	local lmOpt = {}
	lmOpt.rnn_type = rnn_type
	lmOpt.rnn_size = rnn_size
	lmOpt.vocab_size = vocab_size
	lmOpt.word_encoding_size = word_encoding_size
	lmOpt.visual_encoding_size = self.use_dif > 0 and visual_encoding_size*2 or visual_encoding_size
	lmOpt.seq_length = seq_length
	lmOpt.dropout = word_dropout
	self.lm = nn.LanguageModel(self.we, lmOpt)
	-- construct other nonparameterized layer
	self.expander = nn.FeatExpander(seq_per_ref)
end

function layer:training()
	self.we:training()
	self.jemb:training()
	self.lm:training()
end
function layer:evaluate()
	self.we:evaluate()
	self.jemb:evaluate()
	self.lm:evaluate()
end
function layer:getJembParameters()
	local params, grad_params = {}, {}
	-- jemb params
	local p1, g1 = self.jemb:parameters()
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(g1) do table.insert(grad_params, v) end
  return self.flatten(params), self.flatten(grad_params)
end
function layer:getLMParameters()
	-- we don't return jemb parameters, as it's to be finetuned
	-- so the learning rate is difference
	local p1, g1 = self.lm:parameters()
	local p2, g2 = self.we:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  return self.flatten(params), self.flatten(grad_params)
end
function layer:resetExpandSize(seq_per_ref)
	self.expander:resetExpandSize(seq_per_ref)
end
--[[
Inputs:
- feats:
	- cxt_feats 		: N x 4096
	- ann_feats 		: N x 4096
	- lfeats 				: N x 5
	- dif_ann_feats : N x 4096
	- dif_lfeats  	: N x 25
- labels 					: D x (N x seq_per_ref)
Output:
- logprobs				: (D+1) x Ns x (M+1)
where Ns = N x seq_per_ref
]]
function layer:updateOutput(input)
	assert(#input == 2, 'input shoud contain feats and labels')
	assert(#input[1] == 5, 'input[1] should contain cxt_feats, ann_feats, lfeats and dif_feats.')

	local cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats = unpack(input[1])
	local labels = input[2]
	local N = cxt_feats:size(1)
	local seq_per_ref = labels:size(2)/N
	self:resetExpandSize(seq_per_ref)

	-- visual input + difference + seq
	self.visual_input = {ann_feats, lfeats}
	if self.use_context > 0 then table.insert(self.visual_input, 1, cxt_feats) end
	if self.use_dif > 0     then table.insert(self.visual_input, dif_ann_feats); table.insert(self.visual_input, dif_lfeats) end
	self.seq = labels

	-- forward
	self.jemb_feats = self.jemb:forward(self.visual_input)
	self.expanded_feats = self.expander:forward(self.jemb_feats)
	self.logprobs, self.houts = unpack(self.lm:forward{self.expanded_feats, self.seq}) --(seq_length, Ns, Mp1)

	-- return
	self.output = {self.logprobs, self.houts}
	return self.output
end

function layer:updateGradInput(input, gradOutput)
	-- dlogprobs: (seq_length, Ns, Mp1)
	-- dhouts:    (Ns, rnn_size)
	assert(#gradOutput == 2)
	local dexpanded_feats, _ = unpack(self.lm:backward({self.expanded_feats, self.seq}, gradOutput))
	local djemb_feats = self.expander:backward(self.jemb_feats, dexpanded_feats)
	local dvisual_input = self.jemb:backward(self.visual_input, djemb_feats)
	-- we don't bother return
end
--[[
Input:
- feats:
	- cxt_feats 				 : N x 4096
	- ann_feats 				 : N x 4096
	- lfeats   				   : N x 5
	- dif_ann_feats input: N x 4096
	- dif_lfeats 				 : N x 25
Output:
- seq: D x N
]]
function layer:sample(input, opt)
	local sample_max = utils.getopt(opt, 'sample_max', 1)
	local temperature = utils.getopt(opt, 'temperature', 1.0)
	local beam_size = utils.getopt(opt, 'beam_size', 1)  -- if > 1, then do beam-search

	-- visual input + difference + seq
	local cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats = unpack(input)
	local visual_input = {ann_feats, lfeats}
	if self.use_context > 0 then table.insert(visual_input, 1, cxt_feats) end
	if self.use_dif > 0     then table.insert(visual_input, dif_ann_feats); table.insert(visual_input, dif_lfeats) end

	-- sample
	local jemb_feats = self.jemb:forward(visual_input)
	local seq = self.lm:sample(jemb_feats, {sample_max=sample_max, temperature=temperature, beam_size=beam_size})
	return seq
end

