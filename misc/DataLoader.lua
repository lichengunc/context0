require 'hdf5'
local utils = require 'misc.utils'

-- Read json file has
-- 0. refs: list of {ref_id, ann_id, image_id, split, category_id, sent_ids}
-- 1. images: list of {image_id, ref_ids, file_name, width, height, h5_id}
-- 2. anns: list of {ann_id, category_id, image_id, box, h5_id}
-- 3. sentences: list of {sent_id, tokens, h5_id}
-- 4. ix_to_word
-- 5. word_to_ix
-- Read h5 file has
-- /images is (N, 3, 256, 256) uint8 array of raw image in RGB format
-- /labels is (M, max_length) uint32 array of encoded labels, zeros padded

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	-- load the json file which contains info about the dataset
	print('DataLoader loading data.json: ', opt.data_json)
	self.info = utils.read_json(opt.data_json)
	self.ix_to_word = self.info.ix_to_word
	self.vocab_size = utils.count_keys(self.ix_to_word)
	print('vocab size is ' .. self.vocab_size)
	self.images = self.info.images
	self.anns = self.info.anns
	print('We have ' .. #self.images .. ' images.')
	print('We have ' .. #self.anns .. ' anns.')

	-- construct Refs, Images, Anns, and Sentences
	local Refs, Images, Anns, Sentences = {}, {}, {}, {}
	for i, ref in ipairs(self.info.refs) do Refs[ref['ref_id']] = ref end
	for i, image in ipairs(self.info.images) do Images[image['image_id']] = image end
	for i, ann in ipairs(self.info.anns) do Anns[ann['ann_id']] = ann end
	for i, sent in ipairs(self.info.sentences) do Sentences[sent['sent_id']] = sent end
	self.Refs, self.Images, self.Anns, self.Sentences = Refs, Images, Anns, Sentences

	-- construct ann-->ref
	self.annToRef = {}
	for i, ref in ipairs(self.info.refs) do self.annToRef[ref['ann_id']] = ref end

	-- set up some tools
	self.normalizer = nn.Normalize(2):float()
	self.softmaxer  = nn.SoftMax():float()

	-- open hdf5 file
	print('DataLoader loading data.h5: ', opt.data_h5)
	self.data_h5 = hdf5.open(opt.data_h5, 'r')

	local images_size = self.data_h5:read('/images'):dataspaceSize()
	self.max_image_size = images_size[3]
	print('images size is ', images_size)

	local seq_size = self.data_h5:read('/labels'):dataspaceSize()
	self.seq_length = seq_size[2]
	self.num_seqs = seq_size[1]
	print('max sequence length in data is ' .. self.seq_length)

	-- split
	self.split_ix = {}
	self.iterators = {}
	for image_id, image in pairs(self.Images) do  
		local ref_ids = image['ref_ids']		-- we don't bother taking care of google's split, so all ref_ids are within the same split
		local split = self.Refs[ref_ids[1]]['split']  -- we use the first ref's split as image's split
		if not self.split_ix[split] then
			self.split_ix[split] = {}
			self.iterators[split] = 1
		end
		table.insert(self.split_ix[split], image_id)
	end
	for k, v in pairs(self.split_ix) do
		print(string.format('assigned %d images to split %s.', #v, k))
	end
end

function DataLoader:loadFeats(opt)
	-- load the feats.h5 containing /img_feats and /ann_feats
	print('DataLoader loading feats.h5: ', opt.feats_h5)
	self.feats_h5 = hdf5.open(opt.feats_h5, 'r')
end

function DataLoader:loadFeats(opt)
	-- load the feats.h5 containing /img_feats and /ann_feats
	print('DataLoader loading feats.h5: ', opt.feats_h5)
	self.feats_h5 = hdf5.open(opt.feats_h5, 'r')
end

function DataLoader:shuffle(split)
	local iterations = #self.split_ix[split]
	for i = iterations, 2, -1 do
		j = math.random(i)
		self.split_ix[split][i], self.split_ix[split][j] = self.split_ix[split][j], self.split_ix[split][i]
	end
end

function DataLoader:resetIterator(split)
	self.iterators[split] = 1
end

function DataLoader:getVocabSize()
	return self.vocab_size
end

function DataLoader:getVocab()
	return self.ix_to_word
end

function DataLoader:getSeqLength()
	return self.seq_length
end

--[[
Return a batch of data
- image_ids: table of N image_ids
- ref_ids:   table of num_refs ref_ids
- feats:
	- cxt_feats: 			(num_refs, 4096)
	- ann_feats: 			(num_refs, 4096)
	- lfeats:    			(num_refs, 5)
	- dif_ann_feats:  (num_refs, 2096)
	- dif_lfeats: 		(num_refs, 25)
- labels: 					(seq_length, num_refs x seq_per_ref) 
- Rixs: table of [num_refs] LongTensor indicating ref relations, 
				e.g., {torch.longTensor{2, 4}} means 2nd & 4th ref is within the same type of 1st ref
- Lixs: table of [num_refs x seq_per_ref] LongTensor indicating language relations
]]
function DataLoader:getBatch(opt)
	-- general option
	local split = utils.getopt(opt, 'split')
	local batch_size = utils.getopt(opt, 'batch_size', 5)   -- how many ref get returned at once
	local seq_per_ref = utils.getopt(opt, 'seq_per_ref', 3) -- how may sequences to return per ref
	-- ref feat option
	local use_context = utils.getopt(opt, 'use_context', 1) -- 0.none, 1.img, 2.window2, 3.window3
	local use_ann = utils.getopt(opt, 'use_ann', 1)         -- 1.vgg, 2.att, 3.frc
	-- dif feat option
	local use_dif = utils.getopt(opt, 'use_dif', 1)  				-- 0.none, 1.mean, 2.max, 3.weighted
	local dif_use_ann = utils.getopt(opt, 'dif_use_ann', 1) -- 1.vgg, 2.att, 3.frc
	local dif_source =utils.getopt(opt, 'dif_source', 1)    -- 1.st_anns, 2.dt_anns, 3.st_anns+dt_anns
	-- split
	local split_ix = self.split_ix[split]
	assert(split_ix, 'split ' .. split .. 'not found.')
	local max_index = #split_ix

	-- fetch ref_ids appeared in this batch of images
	local image_ids = {}
	local ref_ids = {}
	local ann_ids = {}
	for i=1, batch_size do
		-- fetch image_id
		local ri = self.iterators[split]
		local ri_next = ri+1
		if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
		self.iterators[split] = ri_next
		local image_id = split_ix[ri]
		assert(image_id ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)    
		-- insert image_ids and num_refs
		table.insert(image_ids, image_id)
		-- related refs and anns
		for _, ref_id in ipairs(self.Images[image_id]['ref_ids']) do
			table.insert(ref_ids, ref_id)
			table.insert(ann_ids, self.Refs[ref_id]['ann_id'])
		end
	end

	-- load the feats and labels
	local num_refs = #ref_ids
	local cxt_feats = torch.FloatTensor(num_refs, 4096)
	local ann_feats = torch.FloatTensor(num_refs, 4096)
	local lfeats = torch.FloatTensor(num_refs, 5)
	local dif_ann_feats = torch.FloatTensor(num_refs, 4096)
	local dif_lfeats = torch.FloatTensor(num_refs, 25)
	local label_batch = torch.LongTensor(num_refs*seq_per_ref, self.seq_length)
	local featOpt = {use_context=use_context, use_ann=use_ann, use_dif=use_dif, dif_use_ann=dif_use_ann, dif_source=dif_source}
	for i = 1, num_refs do
		local ref_id = ref_ids[i]
		local ref_ann_id = self.Refs[ref_id]['ann_id']
		cxt_feats[i], ann_feats[i], lfeats[i], dif_ann_feats[i], dif_lfeats[i] = self:fetch_all_feat(ref_ann_id, featOpt)

		-- fetch the labels from ref_id
		local seq = self:fetch_label(ref_id, seq_per_ref)
		local il = (i-1)*seq_per_ref + 1
		label_batch[{ {il, il+seq_per_ref-1} }] = seq
	end

	-- make Rixs
	local Rixs = {}
	local Rn = 0
	for _, image_id in ipairs(image_ids) do
		local image_ref_ids = self.Images[image_id]['ref_ids']
		for i, ref_id in ipairs(self.Images[image_id]['ref_ids']) do
			local rixs = self:fetch_st_ixs(ref_id, image_ref_ids)
			rixs:add(Rn)  -- offset, note it does nothing for empty rixs
			table.insert(Rixs, rixs)
		end
		Rn = Rn + #image_ref_ids
	end
	-- make Lixs
	local Tixs = {}
	for i, rixs in ipairs(Rixs) do
		for j = 1, seq_per_ref do
			local tixs = (rixs-1)*seq_per_ref + j
			Tixs[#Tixs+1] = tixs
		end
	end
	-- return data
	local data = {}
	data.image_ids = image_ids
	data.ref_ids = ref_ids
	data.feats = {cxt_feats, ann_feats, lfeats, dif_ann_feats, dif_lfeats}
	data.labels = label_batch:transpose(1,2):contiguous()
	data.Rixs = Rixs
	data.Tixs = Tixs
	data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
	return data
end

--[[
Find the ixs of cand_ref_ids that share the same category_id of ref_id
]]
function DataLoader:fetch_st_ixs(ref_id, cand_ref_ids)
	local ixs = {}
	for i, cand_ref_id in ipairs(cand_ref_ids) do
		if cand_ref_id ~= ref_id and self.Refs[cand_ref_id]['category_id'] == self.Refs[ref_id]['category_id'] then
			table.insert(ixs, i)
		end
	end
	return torch.LongTensor(ixs)
end

-- return cxt_feat, ann_feat, lfeat, dif_ann_feat, dif_lfeat
function DataLoader:fetch_all_feat(ref_ann_id, opt)
	local use_context = utils.getopt(opt, 'use_context', 1)  -- 0.none, 1.img, 2.window3, 3.window5
	local use_ann     = utils.getopt(opt, 'use_ann', 1) -- 1.vgg, 2.att, 3.frc
	local use_dif = utils.getopt(opt, 'use_dif', 1)  -- 0.none, 1.mean, 2.max, 3.weighted
	local dif_use_ann = utils.getopt(opt, 'dif_use_ann', 1)  -- 1.vgg, 2.att, 3.frc
	local dif_source = utils.getopt(opt, 'dif_source', 1)  -- 1.st_anns, 2.dt_anns, 3.st_anns+dt_anns
	local cxt_feat, ann_feat, lfeat = self:fetch_feat(ref_ann_id, {use_context=use_context, use_ann=use_ann})
	local dif_ann_feat, dif_lfeat = self:fetch_dif_feat(ref_ann_id, {use_dif=use_dif, dif_use_ann=dif_use_ann, dif_source=dif_source})
	return cxt_feat, ann_feat, lfeat, dif_ann_feat, dif_lfeat
end
-- return cxt_feat, ann_feat, lfeat according to the type defined by "use_context" and "use_ann"
function DataLoader:fetch_feat(ann_id, opt)
	local use_context = utils.getopt(opt, 'use_context', 1)  -- 0.none, 1.img, 2.window3, 3.window5
	local use_ann     = utils.getopt(opt, 'use_ann', 1) -- 1.vgg, 2.att, 3.frc
	local ann = self.Anns[ann_id]
	local image = self.Images[ann['image_id']]
	-- fetch cxt_feat
	local cxt_feat = torch.FloatTensor(1, 4096)
	if use_context == 1 then cxt_feat = self.feats_h5:read('/img_feats'):partial(image['h5_id'], {1, 4096}) end
	if use_context == 2 then cxt_feat = self.feats_h5:read('/window2_feats'):partial(ann['h5_id'], {1, 4096}) end
	if use_context == 3 then cxt_feat = self.feats_h5:read('/window3_feats'):partial(ann['h5_id'], {1, 4096}) end
	-- fetch ann_feat
	local ann_feat = torch.FloatTensor(1, 4096)
	if use_ann == 1 then ann_feat = self.feats_h5:read('/ann_feats'):partial(ann['h5_id'], {1, 4096}) end
	if use_ann == 2 then ann_feat = self.feats_h5:read('/att_feats'):partial(ann['h5_id'], {1, 4096}) end
	if use_ann == 3 then ann_feat = self.feats_h5:read('/frc_feats'):partial(ann['h5_id'], {1, 4096}) end
	-- compute lfeat
	local x, y, w, h = unpack(ann['box'])
	local iw, ih = image['width'], image['height']
	local lfeat = torch.FloatTensor{ x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih) }
	-- return
	return cxt_feat, ann_feat, lfeat 
end
--[[
if there is same-category ann_id, we aggregate its difference feature.
if not, we encode with zeros.
For relative location feature, we encode the whole image with binary mask, and take out region centered by object
or we encode relative location of the nearest 5 objects using 20-D vector. Zeros are padded if any.
]]
function DataLoader:fetch_dif_feat(ref_ann_id, opt)
	local use_dif = utils.getopt(opt, 'use_dif', 1)  -- 0.none, 1.mean, 2.max, 3.weighted
	local dif_use_ann = utils.getopt(opt, 'dif_use_ann', 1)  -- 1.vgg, 2.att, 3.frc
	local dif_max = utils.getopt(opt, 'dif_max', 5)  -- max num of relative objects
	local dif_source = utils.getopt(opt, 'dif_source', 1)  -- 1.st_anns, 2.dt_anns, 3.st_anns+dt_anns

	local dif_ann_feat = torch.FloatTensor(4096):zero()
	local dif_lfeat    = torch.FloatTensor(dif_max*5):zero()
	local _, st_ann_ids, _, dt_ann_ids = self:fetch_neighbour_ids(ref_ann_id)
	local cand_ann_ids
	if dif_source == 1 then cand_ann_ids = st_ann_ids end
	if dif_source == 2 then cand_ann_ids = dt_ann_ids end
	if dif_source == 3 then cand_ann_ids = utils.combineTable(st_ann_ids, dt_ann_ids) end
	if #cand_ann_ids ~= 0 then -- if no nearby same-type object is found, return zeros
		-- get cand_ann_feats
		if #cand_ann_ids > dif_max then for k = dif_max+1, #cand_ann_ids do table.remove(cand_ann_ids, k) end end
		local cand_ann_feats = torch.FloatTensor(#cand_ann_ids, 4096)
		for j, cand_ann_id in ipairs(cand_ann_ids) do
			_, cand_ann_feats[j], _ = self:fetch_feat(cand_ann_id, {use_context=0, use_ann=dif_use_ann}) 
		end
		-- get ref_ann_feat
		local _, ref_ann_feat, _ = self:fetch_feat(ref_ann_id, {use_context=0, use_ann=dif_use_ann}) 
		-- compute pooled feat
		if use_dif == 1 then -- mean
			cand_ann_feats = cand_ann_feats - ref_ann_feat:expandAs(cand_ann_feats)
			dif_ann_feat = torch.mean(cand_ann_feats, 1) 
		elseif use_dif == 2 then -- max
			cand_ann_feats = cand_ann_feats - ref_ann_feat:expandAs(cand_ann_feats)
			dif_ann_feat = torch.max(cand_ann_feats, 1)
		elseif use_dif == 3 then -- weighted
			local weights = self:computeWeights(ref_ann_feat, cand_ann_feats, ref_ann_id, cand_ann_ids)
			for q = 1, #cand_ann_ids do
				dif_ann_feat = dif_ann_feat + (cand_ann_feats[q] - ref_ann_feat)*weights[q]
			end
		elseif use_dif == 4 then  -- min
			cand_ann_feats = cand_ann_feats - ref_ann_feat:expandAs(cand_ann_feats)
			dif_ann_feat = torch.min(cand_ann_feats, 1)
		end
		-- compute dif_lfeat
		local image = self.Images[self.Anns[ref_ann_id]['image_id']]
		local iw, ih = image['width'], image['height']
		local rbox = self.Anns[ref_ann_id]['box']
		local rcx, rcy, rw, rh = rbox[1]+rbox[3]/2, rbox[2]+rbox[4]/2, rbox[3], rbox[4]
		for j = 1, math.min(5, #cand_ann_ids) do
			local cbox = self.Anns[cand_ann_ids[j]]['box']
			local ccx, ccy, cw, ch = cbox[1]+cbox[3]/2, cbox[2]+cbox[4]/2, cbox[3], cbox[4]
			dif_lfeat[{ {(j-1)*5+1, j*5} }] = torch.FloatTensor{ ccx-rcx, ccy-rcy, ccx+cw-rcx, ccy+ch-rcy, cw*ch/(rw*rh) } -- we don't bother normalizing here.
		end
	end

	return dif_ann_feat, dif_lfeat
end
-- compute weights, sum as 1
function DataLoader:computeWeights(ref_ann_feat, cand_ann_feats, ref_ann_id, cand_ann_ids)
	local function cossim(A, b) 
		return torch.mv(self.normalizer:forward(A), self.normalizer:forward(b):view(-1))
	end
	if #cand_ann_ids == 1 then return torch.FloatTensor{1} end
	local weights = torch.FloatTensor(#cand_ann_ids)
	local image = self.Images[self.Anns[ref_ann_id]['image_id']]
	local iw, ih = image['width'], image['height']
	local rbox = self.Anns[ref_ann_id]['box']
	local rcx, rcy, rw, rh = rbox[1]+rbox[3]/2, rbox[2]+rbox[4]/2, rbox[3], rbox[4]
	for i, cand_ann_id in ipairs(cand_ann_ids) do
		local cbox = self.Anns[cand_ann_id]['box']
		local ccx, ccy, cw, ch = cbox[1]+cbox[3]/2, cbox[2]+cbox[4]/2, cbox[3], cbox[4]
		weights[i] = - math.abs((ccx-rcx)/iw) - math.abs((ccy-rcy)/ih) + math.abs(cw*ch/(iw*ih))
	end
	weights:add(cossim(cand_ann_feats, ref_ann_feat))
	-- weights:div(torch.sum(weights))
	-- weights:cmul(cossim(cand_ann_feats, ref_ann_feat))
	weights = self.softmaxer:forward(weights)
	return weights
end
--[[
For given ref_ann_id, we return
- st_ann_ids: same-type neighbouring ann_ids (not including itself)
- dt_ann_ids: different-type neighbouring ann_ids
Ordered by distance to the input ann_id
]]
function DataLoader:fetch_neighbour_ids(ref_ann_id)
	local ref_ann = self.Anns[ref_ann_id]
	local x, y, w, h = unpack(ref_ann['box'])
	local rx, ry = x+w/2, y+h/2
	local function compare(ann_id1, ann_id2) 
		local x, y, w, h = unpack(self.Anns[ann_id1]['box'])
		local ax1, ay1 = x+w/2, y+h/2
		local x, y, w, h = unpack(self.Anns[ann_id2]['box'])
		local ax2, ay2 = x+w/2, y+h/2
		return (rx-ax1)^2 + (ry-ay1)^2 < (rx-ax2)^2 + (ry-ay2)^2  -- closer --> former
	end
	local image = self.Images[ref_ann['image_id']]
	local ann_ids = utils.copyTable(image['ann_ids'])
	table.sort(ann_ids, compare)

	local st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = {}, {}, {}, {}
	for i = 1, #ann_ids do
		local ann_id = ann_ids[i]
		if ann_id ~= ref_ann_id then
			if self.Anns[ann_id]['category_id'] == ref_ann['category_id'] then
				table.insert(st_ann_ids, ann_id)
				if self.annToRef[ann_id] ~= nil then table.insert(st_ref_ids, self.annToRef[ann_id]['ref_id']) end 
			else
				table.insert(dt_ann_ids, ann_id)
				if self.annToRef[ann_id] ~= nil then table.insert(dt_ref_ids, self.annToRef[ann_id]['ref_id']) end
			end
		end
	end
	return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids
end

function DataLoader:fetch_label(ref_id, num_sents)
	local ref = self.Refs[ref_id]
	local sent_ids = ref['sent_ids']
	local seq = torch.LongTensor(num_sents, self.seq_length)
	if #sent_ids < num_sents then
		for q = 1, num_sents do
			local ix = torch.random(1, #sent_ids)
			local sent_id = sent_ids[ix]
			local sent_h5_id = self.Sentences[sent_id]['h5_id']
			seq[q] = self.data_h5:read('/labels'):partial({sent_h5_id, sent_h5_id}, {1, self.seq_length})
		end
	else
		local ix = torch.random(1, #sent_ids-num_sents+1)  -- pick up the start ix
		local ixs = {}
		for q = 1, num_sents do table.insert(ixs, ix+q-1) end  
		ixs = utils.shuffleTable(ixs)  -- shuffle ixs
		for q, ix in ipairs(ixs) do
			local sent_id = sent_ids[ix]
			local sent_h5_id = self.Sentences[sent_id]['h5_id']
			seq[q] = self.data_h5:read('/labels'):partial({sent_h5_id, sent_h5_id}, {1, self.seq_length})
		end
	end
	return seq
end