package.path = '../?.lua;' .. package.path

require 'torch'
require 'nn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
require 'misc.DataLoader'
local utils = require 'misc.utils'
-- local net_utils = require 'misc.net_utils'


function decode_sequence(ix_to_word, seq)
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

local data_json = '../cache/data/' .. 'refcoco+_unc' .. '/data.json'  -- path to the json file containing additional info and vocab
local data_h5   = '../cache/data/' .. 'refcoco+_unc' .. '/data.h5'    -- path to the h5file containing the preprocessed dataset
local feats_h5 = '../cache/data/' .. 'refcoco+_unc' ..'/feats.h5' 
local loader = DataLoader{data_h5 = data_h5, data_json = data_json}
loader:loadFeats{feats_h5 = feats_h5}
local ix_to_word = loader.ix_to_word

for i = 1, 1 do

	local data = loader:getBatch{split='val', batch_size=1, seq_per_ref=3}
	local data = loader:getBatch{split='val', batch_size=1, seq_per_ref=3}
	local data = loader:getBatch{split='val', batch_size=1, seq_per_ref=3}
	local data = loader:getBatch{split='val', batch_size=1, seq_per_ref=3}
	local data = loader:getBatch{split='val', batch_size=1, seq_per_ref=3}
	print(data.ref_ids)
	print(data.Rixs)

	local p = 1
	print(loader.Refs[data.ref_ids[p]].category_id)
	local rixs = data.Rixs[p]
	if rixs:nElement() > 0 then
		for i=1, rixs:nElement() do
			local rix = rixs[i]
			print(loader.Refs[data.ref_ids[rix]].category_id)
		end
	end
	print('\n')

	-- check the correspondance between Rixs and Tixs
	print(data.Rixs[2])
	print(data.Tixs[4], data.Tixs[5], data.Tixs[6])

	local q = 6
	local tixs = data.Tixs[q]
	print(q, tixs)
	local sents = data.labels[{ {}, {q} }]
	print(decode_sequence(ix_to_word, sents))
	local sents = data.labels:index(2, tixs)
	print(decode_sequence(ix_to_word, sents))



end

-- for i = 1,20 do
-- 	-- fetch ref objects
-- 	local data = loader:getBatch({split='val', batch_size=1, seq_per_ref=3})
-- 	local ref_ids = data.ref_ids
-- 	local ref_ann_ids = data.ref_ann_ids
-- 	local infos = data.infos
-- 	local feats = data.feats
-- 	-- print(feats)
-- 	-- print(infos)
-- 	-- fetch negative samples
-- 	local neg_data = loader:getNegativeBatch(ref_ann_ids, {split='val', batch_size=1, seq_per_ref=3})
-- 	-- print positive and negative labels
-- 	local seq = net_utils.decode_sequence(ix_to_word, data.labels)
-- 	print(seq)
-- 	local seq = net_utils.decode_sequence(ix_to_word, neg_data.neg_labels)
-- 	print(seq)
-- 	print('\n')
-- end
