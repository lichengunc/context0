require 'image'
require 'nngraph'
require 'misc.basic_modules'
local utils = require 'misc.utils'
local net_utils = {}

--[[
-- visual input encoding layer
Input:
- cxt_feats: (n, 4096)
- ann_feats: (n, 4096)
- lfeats:    (n, 5)
Output:
- jemb_feats:(n, 512)
]]
function net_utils.build_visual_encoder(cnn, opt)
	local visual_encoding_size = utils.getopt(opt, 'visual_encoding_size', 512)
	local dropout = utils.getopt(opt, 'dropout', 0)
	local use_context = utils.getopt(opt, 'use_context', 1)  -- 0.none, 1.img, 2.window
	local init_norm = utils.getopt(opt, 'init_norm', 10)
	local fc8 = cnn:get(39)  -- fc8 layer
  local scale_layer1 = nn.CMul(1000); scale_layer1.weight:fill(init_norm)
  local scale_layer2 = nn.CMul(5);    scale_layer2.weight:fill(init_norm)

  local M = nn.ParallelTable()
  -- context path
  if use_context > 0 then
    local cxt_view = nn.Sequential()
    cxt_view:add(fc8:clone())
    cxt_view:add(nn.Normalize(2))
    cxt_view:add(scale_layer1:clone())
    M:add(cxt_view)
  end
  -- region path
  local ann_view = nn.Sequential()
  ann_view:add(fc8:clone())
  ann_view:add(nn.Normalize(2))
  ann_view:add(scale_layer1:clone())  -- Be careful, we don't wanna share parameters here!
  M:add(ann_view)
  -- location path
	local loc_view = nn.Sequential()
	loc_view:add(nn.Normalize(2))
	loc_view:add(scale_layer2)
	M:add(loc_view)
	-- jemb
	local jemb = nn.Sequential()
	jemb:add(M)
	jemb:add(nn.JoinTable(2))
	local d = 1005
	if use_context > 0 then d = d+1000 end
	jemb:add(nn.Linear(d, visual_encoding_size))
	jemb:add(nn.Dropout(dropout))
	return jemb
end
--[[
visual difference encoding layer
Input:
- dif_ann_feats: (n, 4096)
- dif_lfeats:    (n, 5)
]]
function net_utils.build_dif_encoder(cnn, opt)
	local visual_encoding_size = utils.getopt(opt, 'visual_encoding_size', 512)
  local dropout = utils.getopt(opt, 'dropout', 0)
  local init_norm = utils.getopt(opt, 'init_norm', 10)
  local fc8 = cnn:get(39)  -- fc8 layer
  local scale_layer1 = nn.CMul(1000); scale_layer1.weight:fill(init_norm)
  local scale_layer2 = nn.CMul(25);   scale_layer2.weight:fill(init_norm)

  local M = nn.ParallelTable()
  -- visual difference
  local dif_ann_view = nn.Sequential()
  dif_ann_view:add(fc8)  -- reduce dimension
  dif_ann_view:add(nn.Normalize(2)); 
  dif_ann_view:add(scale_layer1)
  M:add(dif_ann_view)
  -- location difference
  local dif_loc_view = nn.Sequential()
  dif_loc_view:add(nn.Normalize(2))
  dif_loc_view:add(scale_layer2)
  M:add(dif_loc_view)
  -- jemb
  local jemb = nn.Sequential()
  jemb:add(M)
  jemb:add(nn.JoinTable(2))
  jemb:add(nn.Linear(1025, visual_encoding_size))
  jemb:add(nn.Dropout(dropout))
  return jemb
end
--[[
Inputs:
- feats:
	- cxt_feats     : N x 4096
	- ann_feats 		: N x 4096
	- lfeats 				: N x 5
	- dif_ann_feats : N x 4096
	- dif_lfeats  	: N x 25
Output:
- encoded_feats   : N x (512+512)
]]
function net_utils.build_jemb(cnn1, cnn2, opt)
  local use_context = utils.getopt(opt, 'use_context')
  local use_dif = utils.getopt(opt, 'use_dif')
  local init_norm = utils.getopt(opt, 'init_norm')
  local visual_encoding_size = utils.getopt(opt, 'visual_encoding_size')
  local dropout = utils.getopt(opt, 'dropout')
  -- make inputs
  local inputs = {}
  local num_inputs, D = 2, 1005  -- we must have ann_feats and lfeats
  if use_context > 0 then num_inputs = num_inputs+1; D = D+1000 end
  if use_dif > 0 then num_inputs = num_inputs+2; D = D+1025 end
  for k = 1,num_inputs do table.insert(inputs, nn.Identity()()) end
  -- representation of the referred object
  local visual_encoder = net_utils.build_visual_encoder(cnn1, {use_context=use_context, init_norm=init_norm, 
  	visual_encoding_size=visual_encoding_size})
  local visual_input = {}
  if use_context == 0 then visual_input = {inputs[1], inputs[2]} end
  if use_context > 0 then visual_input = {inputs[1], inputs[2], inputs[3]} end
  local f1 = visual_encoder(visual_input)
  -- representation of visual difference
  local dif_input, dif_encoder, f2
  if use_dif > 0 then
    dif_input = {inputs[#inputs-1], inputs[#inputs]}
    dif_encoder = net_utils.build_dif_encoder(cnn2, {init_norm=init_norm, visual_encoding_size=visual_encoding_size, dropout=dropout})
    f2 = dif_encoder(dif_input)
  end
  -- join the two type of features
  local output
  if use_dif == 0 then output = f1 else output = nn.JoinTable(2)({f1, f2}) end
  -- return
  return nn.gModule(inputs, {output})
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end
--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

function net_utils.language_eval(predictions, dataset, id, split)
  local cache_lang_dataset_dir = path.join('./cache/lang', dataset)
  -- we don't check isdir here...
  -- otherwise we have to luarocks install some other packages, e.g., posix, luafilesystem
  os.execute('mkdir '..cache_lang_dataset_dir)  

  local cache_path = path.join(cache_lang_dataset_dir, 'model_id' .. id .. '_' .. split .. '.json')
  utils.write_json(cache_path, {predictions = predictions})
  -- call python to evaluate each sent with ground-truth sentences
  os.execute('python ./pyutils/python_eval_lang.py ' .. '--dataset_splitBy ' .. dataset .. ' --model_id ' .. id .. ' --split ' .. split)
  -- return results
  local out_path = path.join(cache_lang_dataset_dir, 'model_id' .. id .. '_' .. split .. '_out.json')
  local out = utils.read_json(out_path)
  local result_struct = out['overall']  -- overall scores
  return result_struct
end

return net_utils




