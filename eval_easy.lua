--[[
Before evaluation, call pyutils/generate_gdboxes.py and pyutils/generate_gdsents.py

The required inpus are 
1. imageToBoxes.json: dict of {'image_id': [{'box', 'h5_id'}], here h5_id refers to box_feats.h5 file
2. sents.json: list of {'sent', 'image_id', 'ref_id', 'tokens', 'box', 'split'}
--]]
require 'torch'
require 'nn'
require 'cudnn'
require 'hdf5'
require 'image'
require 'cutorch'
require 'cunn'
-- local imports
require 'misc.basic_modules'
require 'misc.EasyLoader'
require 'misc.full_model'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'


-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test accuracy of click(bbox) given sentence')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset+splitBy')
cmd:option('-id', 0, 'model id to be evaluated')
-- Basic options
cmd:option('-num_sents', -1, 'how many sentences to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-batch_size', 8, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-vis', -1, 'if 1 then we visualize boxes during evaluation')
-- Test on what split
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- Use ground-truth boxes or detected objects(proposals)
cmd:option('-boxes_type', 'gd', 'use ground-truth (gd) or predicted (pred) imageToBoxes?')
-- misc
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()


local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')  -- for CPU
cutorch.setDevice(opt.gpuid + 1)  -- note +1 because lua is 1-indexed

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.dataset) > 0 and string.len(opt.id) > 0, 'must provide dataset name and model id')
model_path = path.join('model', opt.dataset, 'model_id' .. opt.id .. '.t7')
local checkpoint = torch.load(model_path)

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'seq_per_ref', 'use_context', 'use_ann', 'use_dif', 'dif_use_ann', 'dif_source', 'use_location'}
for k, v in pairs(fetch) do
	opt[v] = checkpoint.opt[v]  -- copy over options from model
end

-- load networks from model checkpoint
local protos = checkpoint.protos
if opt.gpuid >= 0 then for k, v in pairs(protos) do v:cuda() end end


-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.dataset) > 0 and string.len(opt.id) > 0, 'must provide dataset name and model id')
model_path = path.join('model', opt.dataset, 'model_id' .. opt.id .. '.t7')
local checkpoint = torch.load(model_path)

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'seq_per_ref', 'use_context', 'use_ann', 'use_dif', 'dif_use_ann', 'dif_source', 'use_location'}
for k, v in pairs(fetch) do
	opt[v] = checkpoint.opt[v]  -- copy over options from model
end

-- load networks from model checkpoint
local protos = checkpoint.protos
if opt.gpuid >= 0 then for k, v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
-- Data information
local data_json = 'cache/data/' .. opt.dataset .. '/data.json'  -- path to the json file containing additional info and vocab
local data_h5 = 'cache/data/' .. opt.dataset .. '/data.h5'    -- path to the h5file containing imgs, labels
local feats_h5 = 'cache/data/' .. opt.dataset .. '/feats.h5'  -- path to the h5file containing img_feats, ann_feats

-- BoxLoader
local loader = EasyLoader{data_json = data_json, data_h5 = data_h5, feats_h5 = feats_h5}

-------------------------------------------------------------------------------
-- Evaluation fun(ction)
-------------------------------------------------------------------------------
protos.full_model:evaluate()

local predictions = {}
local acc = 0
local ix = 1
while true do

	-- fetch next sent
	local dataOpt = {}
	dataOpt.split = opt.split
	dataOpt.use_context = opt.use_context
	dataOpt.use_ann = opt.use_ann
	dataOpt.use_dif = opt.use_dif
	dataOpt.dif_use_ann = opt.dif_use_ann
	dataOpt.dif_source = opt.dif_source
	local data = loader:getBatch(dataOpt)
	if opt.num_sents > 0 and ix > opt.num_sents then break end  -- we've used up opt.num_sents sents

	-- print status
	if opt.num_sents > 0 then 
		print(string.format('Processing sent(%d/%d)', ix, opt.num_sents)) 
	else 
		print(string.format('Processing sent(%d/%d)', ix, loader:getNumSents{split = opt.split}))
	end
	ix = ix+1

	local gd_ix = data.gd_ix

	local feats = data.feats
  -- ship to GPU
  if opt.gpuid >= 0 then
    for k, v in ipairs(feats) do feats[k] = feats[k]:cuda() end
  end

	local labels = data.labels:long()

	-- forward network to compute loss for each candidate box
  local logprobs, _ = unpack(protos.full_model:forward({feats, labels}))
	local losses = computeLosses(logprobs, labels)

	-- which one gets max prob?
	local ls, bix = torch.min(losses, 1)
	if bix[1] == gd_ix then
		acc = acc + 1
	end

	-- add to predictions
	local entry = {sent_id = data.sent_id, image_id = data.image_id,  gd_ann_id = data.ann_ids[gd_ix], pred_ann_id = data.ann_ids[bix[1]]}
	table.insert(predictions, entry)

	if data.wrapped then break end  -- if si_next exceeds #split_ix, break.
end

print('accuracy = ', acc/ix)

-- save results
local cache_box_dataset_dir = path.join('cache/box', opt.dataset)
os.execute('mkdir ' .. cache_box_dataset_dir)
local cache_path = path.join(cache_box_dataset_dir, 'model_id' .. opt.id .. '_' .. opt.split .. '.json')
utils.write_json(cache_path, {predictions=predictions})



