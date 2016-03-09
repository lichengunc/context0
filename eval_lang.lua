require 'torch'
require 'nn'
require 'cudnn'
require 'nngraph'
-- local imports
require 'misc.basic_modules'
require 'misc.DataLoader'
require 'misc.full_model'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Refer Expression model')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset+splitBy')
cmd:option('-id', 0, 'model id to be evaluated')  -- corresponding to opt.id in train.lua
-- Basic options
cmd:option('-batch_size', 8, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- For evaluation on refer dataset for some split
cmd:option('-split', 'testA', 'what split to use: val|test|train')
-- misc
cmd:option('-seed', 24, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

cmd:text()

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.dataset) > 0 and string.len(opt.id) > 0, 'must provide dataset name and model id')
local model_path = path.join('model', opt.dataset, 'model_id' .. opt.id .. '.t7')
local checkpoint = torch.load(model_path)

-- override and collect parameters
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'seq_per_ref', 'use_context', 'use_dif', 'dif_use_ann'}
for k, v in pairs(fetch) do
	opt[v] = checkpoint.opt[v]
end
local vocab = checkpoint.vocab

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local data_h5 = 'cache/data/' .. opt.dataset .. '/data.h5'    -- path to the h5file containing imgs, labels
local data_json = 'cache/data/' .. opt.dataset .. '/data.json'  -- path to the json file containing additional info and vocab
local loader = DataLoader{data_h5 = data_h5, data_json = data_json}

local feats_h5 = 'cache/data/' .. opt.dataset ..'/feats.h5' 
loader:loadFeats{feats_h5 = feats_h5}

-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
-- load jemb, word-embedding and LM from checkpoint
local protos = checkpoint.protos 
-- reconstruct clones inside the language model
protos.full_model.lm:createClones() -- reconstruct clones inside the language model
-- manually create language criterion
protos.crit = nn.LanguageModelCriterion()

-- ship everything to GPU
if opt.gpuid >= 0 then
	for k, v in pairs(protos) do v:cuda() end
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalOpt)

  local verbose = utils.getopt(evalOpt, 'verbose', true)
  local val_images_use = utils.getopt(evalOpt, 'num_images')

  protos.full_model:evaluate()

  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

  	local dataOpt = {}
  	dataOpt.split = split
  	dataOpt.batch_size = opt.batch_size
  	dataOpt.seq_per_ref = opt.seq_per_ref
  	dataOpt.use_context = opt.use_context
  	dataOpt.use_ann = opt.use_ann
    dataOpt.use_dif = opt.use_dif
    dataOpt.dif_use_ann = opt.dif_use_ann
    dataOpt.dif_source = opt.dif_source
    local data = loader:getBatch(dataOpt)
    n = n + opt.batch_size    

    -- visual input
    local feats = data.feats
    local labels = data.labels

    -- ship to gpud
    if opt.gpuid >= 0 then
    	for k,v in ipairs(feats) do feats[k] = feats[k]:cuda() end
    end

    -- forward pass
    local logprobs, houts = unpack(protos.full_model:forward{feats, labels})
    local loss = protos.crit:forward(logprobs, labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to get generated samples for each ref
    local sampleOpts = {}
    sampleOpts.sample_max = opt.sample_max
    sampleOpts.beam_size = opt.beam_size
    sampleOpts.temperature = opt.temperature
    local seq = protos.full_model:sample(feats, sampleOpts)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1, #sents do
      local entry = {ref_id = data.ref_ids[k], sent = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('ref_id%s: %s', entry.ref_id, entry.sent))
      end
    end

    -- if we wrapped around the split or used up val imgs budget than bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 500 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if val_images_use >= 0 and n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.dataset, opt.id, split)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

local loss, split_predictions, lang_stats = eval_split(opt.split, {num_images = opt.num_images})
print('loss: ', loss)
if lang_stats then
  print(lang_stats)
end
-- write HTML, feed in dataset, model_id, split
-- saved in vis/dataset/
-- os.execute('python vis.py ' .. '--dataset_splitBy_click ' .. opt.dataset .. ' --model_id ' .. opt.id .. ' --split ' .. opt.split) 
os.execute('python vis_lang.py ' .. '--dataset_splitBy ' .. opt.dataset .. ' --model_id ' .. opt.id .. ' --split ' .. opt.split) 


