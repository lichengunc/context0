require 'torch'
require 'nn'
require 'cudnn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
require 'misc.DataLoader'
require 'misc.full_model'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local basic_modules = require 'misc.basic_modules'
require 'misc.optim_updates'


-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-dataset', 'refcoco_unc', 'name of our our dataset+splitBy')
cmd:option('-feats_type', 1, '1 from raw image; 2 from h5 image')
cmd:option('-cnn_proto', 'model/vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model', 'model/vgg/VGG_ILSVRC_16_layers.caffemodel', 'path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- lm setting
cmd:option('-rnn_size', 'lstm', 'lstm or gru')
cmd:option('-rnn_size', 512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-word_encoding_size', 512,'the encoding size of each token in the vocabulary.')
cmd:option('-word_dropout', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-seq_per_ref', 3,'number of captions to sample for each ref during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- visual input encoding
cmd:option('-visual_encoding_size', 512, 'visual encoding size')
cmd:option('-visual_dropout', 0.2, 'strength of dropout in the visual input')
cmd:option('-init_norm', 10, 'normalization weight')
cmd:option('-use_context',  1, '0.none / 1.img / 2.window2 / 3.window3')
cmd:option('-use_ann',      1, '1.vgg  / 2.att / 3.frc')
cmd:option('-use_dif',      0, '0.none / 1.mean / 2.max / 3.weighted / 4.min')
cmd:option('-dif_use_ann',  1, '1.vgg  / 2.att / 3.frc')
cmd:option('-dif_source',   1, '1.st_anns / 2.dt_anns / 3.st_anns+dt_anns')

-- ranking setting
cmd:option('-ranking_weight1', 0, 'the weight on ranking loss over objects')
cmd:option('-ranking_weight2', 0, 'the weight on ranking loss over sentences')
cmd:option('-ranking_margin', 1, 'the margin in the ranking loss')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', 8, 'what is the batch size in number of images per batch? (there will be x seq_per_ref sentences)')
cmd:option('-grad_clip', 0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-learning_rate_decay_start', 10000, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 20000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_epsilon', 1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for LM model
cmd:option('-lm_optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-lm_learning_rate', 4e-4,'learning rate')
cmd:option('-lm_optim_alpha', 0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-lm_optim_beta', 0.999,'beta used for adam')

-- Optimization: for Joint Embedding
cmd:option('-finetune_jemb_after', 0, 'After what iteration do we start finetuning the joint embedding? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-jemb_optim', 'adam','optimization to use for joint embedding')
cmd:option('-jemb_optim_alpha', 0.8,'alpha for momentum of joint embedding')
cmd:option('-jemb_optim_beta', 0.999,'alpha for momentum of joint embedding')
cmd:option('-jemb_learning_rate', 4e-5,'learning rate for the joint embedding')
cmd:option('-jemb_weight_decay', 0, 'L2 weight decay just for the joint embedding')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', 0, 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 8, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local data_h5   = 'cache/data/' .. opt.dataset .. '/data.h5'    -- path to the h5file containing the preprocessed dataset
local data_json = 'cache/data/' .. opt.dataset .. '/data.json'  -- path to the json file containing additional info and vocab
local loader = DataLoader{data_h5 = data_h5, data_json = data_json}

local feats_h5 = 'cache/data/' .. opt.dataset ..'/feats.h5' 
loader:loadFeats{feats_h5 = feats_h5}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
if string.len(opt.start_from) > 0 then
	local loaded_checkpoint = torch.load(opt.start_from)
	protos = loaded_checkpoint.protos
	net_utils.unsanitize_gradients(protos.full_model.jemb)
	net_utils.unsanitize_gradients(protos.full_model.lm.core)
	net_utils.unsanitize_gradients(protos.full_model.we)
  -- prepare criterions
  protos.crit = nn.LanguageModelCriterion()
else
  -- initialize the ConvNet
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, 'cudnn')
  -- create protos for full model
  local modelOpt = {}
  -- jemb part
  modelOpt.visual_encoding_size = opt.visual_encoding_size
  modelOpt.visual_dropout = opt.visual_dropout
  modelOpt.init_norm = opt.init_norm
  modelOpt.use_context = opt.use_context
  modelOpt.use_dif = opt.use_dif
  -- lm part
  modelOpt.rnn_type = opt.rnn_type
  modelOpt.rnn_size = opt.rnn_size
  modelOpt.vocab_size = loader:getVocabSize()
  modelOpt.seq_length = loader:getSeqLength()
  modelOpt.word_encoding_size = opt.word_encoding_size
  modelOpt.word_dropout = opt.word_dropout
  modelOpt.seq_per_ref = opt.seq_per_ref
  protos.full_model = nn.FullModel(cnn_raw, modelOpt)
  -- prepare criterions
  protos.crit = nn.LanguageModelCriterion()
end

-- ship every thing to GPU
if opt.gpuid >= 0 then
	for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector
local jemb_params, jemb_grad_params = protos.full_model:getJembParameters()
print('total number of parameters in JEMB: ', jemb_params:nElement())
assert(jemb_params:nElement() == jemb_grad_params:nElement())

local lm_params, lm_grad_params = protos.full_model:getLMParameters()
print('total number of parameters in FULL MODEL: ', lm_params:nElement())
assert(lm_params:nElement() == lm_grad_params:nElement())

-- construct thin module clones that share parameters with the actual modules
local thin_full_model = protos.full_model:clone()
thin_full_model.we:share(protos.full_model.we, 'weight')
thin_full_model.lm.core:share(protos.full_model.lm.core, 'weight', 'bias')
thin_full_model.jemb:share(protos.full_model.jemb, 'weight', 'bias')
-- sanitize all modules of gradient storage
net_utils.sanitize_gradients(thin_full_model.we)
net_utils.sanitize_gradients(thin_full_model.lm.core)
net_utils.sanitize_gradients(thin_full_model.jemb)

protos.full_model.lm:createClones()
collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalOpt)

  local verbose = utils.getopt(evalOpt, 'verbose', true)
  local val_images_use = utils.getopt(evalOpt, 'val_images_use')

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
    local seq = protos.full_model:sample(feats)
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

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
-- shuffle the data
loader:shuffle('train')
local iter = 0

local function lossFun()
	protos.full_model:training()
	jemb_grad_params:zero()
	lm_grad_params:zero()

  local dataOpt = {}
  dataOpt.split = 'train'
  dataOpt.batch_size = opt.batch_size
  dataOpt.seq_per_ref = opt.seq_per_ref
  dataOpt.use_context = opt.use_context
  dataOpt.use_ann = opt.use_ann
	dataOpt.use_dif = opt.use_dif
	dataOpt.dif_use_ann = opt.dif_use_ann
	dataOpt.dif_source = opt.dif_source
	local data = loader:getBatch(dataOpt)

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
  -- backward pass
  local dlogprobs = protos.crit:backward(logprobs, labels)
  protos.full_model:backward({feats, labels}, {dlogprobs, houts:clone():zero()})

  -- clip gradients
  jemb_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  lm_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- let's get out!
  local losses = {total_loss = loss}
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local jemb_optim_state = {}
local lm_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

while true do
  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('dataset[%s], id[%s], gpuid[%s], iter %d: %f', opt.dataset, opt.id, opt.gpuid, iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    val_loss_history[iter] = val_loss
    if lang_stats then
      print(lang_stats)
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, opt.dataset, 'model_id' .. opt.id)
    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history
    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.full_model = thin_full_model
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rates for LM and JEMB
  local lm_learning_rate = opt.lm_learning_rate
  local jemb_learning_rate = opt.jemb_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.1, frac)
    lm_learning_rate = lm_learning_rate * decay_factor
    jemb_learning_rate = jemb_learning_rate * decay_factor
  end

  -- perform LM and WE update
  if opt.lm_optim == 'rmsprop' then
    rmsprop(lm_params, lm_grad_params, lm_learning_rate, opt.lm_optim_alpha, opt.optim_epsilon, lm_optim_state)
  elseif opt.lm_optim == 'sgd' then
    sgd(lm_params, lm_grad_params, opt.lm_learning_rate)
  elseif opt.lm_optim == 'adam' then
    adam(lm_params, lm_grad_params, lm_learning_rate, opt.lm_optim_alpha, opt.lm_optim_beta, opt.optim_epsilon, lm_optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a JEMB update (if finetuning, and if rnn above us is not warming up right now)
  if opt.jemb_optim == 'rmsprop' then
    rmsprop(jemb_params, jemb_grad_params, jemb_learning_rate, opt.jemb_optim_alpha, opt.optim_epsilon, jemb_optim_state)
  elseif opt.jemb_optim == 'sgd' then
    sgd(jemb_params, jemb_grad_params, opt.jemb_learning_rate)
  elseif opt.jemb_optim == 'adam' then
    adam(jemb_params, jemb_grad_params, jemb_learning_rate, opt.jemb_optim_alpha, opt.jemb_optim_beta, opt.optim_epsilon, jemb_optim_state)
  else
    error('bad option opt.jemb_optim')
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 500 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end