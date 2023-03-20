class grounding_model_multihop(nn.Module):
    def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1, bert_model='bert-base-uncased', \
        NFilm=2, fusion='prod', intmd=False, mstage=False, convlstm=False, \
        coordmap=True, leaky=False, dataset=None, bert_emb=False, tunebert=False, use_sal=False, use_paf=False):
        super(grounding_model_multihop, self).__init__()
        self.coordmap = coordmap
        self.emb_size = emb_size
        self.NFilm = NFilm
        self.intmd = intmd
        self.mstage = mstage
        self.convlstm = convlstm
        self.tunebert = tunebert
        self.use_sal = use_sal
        self.use_paf = use_paf
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        ## Text model
        self.textmodel = BertModel.from_pretrained(bert_model)
        
        self.mapping_visu = ConvBatchNormReLU(512 if self.convlstm else 256, emb_size, 1, 1, 0, 1, leaky=leaky)
        
        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.ReLU(),)
          
        textdim=emb_size
        
        self.film = FiLMedConvBlock_multihop(NFilm=3,textdim=textdim,visudim=emb_size,\
            emb_size=emb_size,fusion=fusion,intmd=(intmd or mstage or convlstm))
            
        self.film1 = FiLMedConvBlock_multihop(NFilm=1,textdim=textdim,visudim=emb_size,\
            emb_size=emb_size,fusion=fusion,intmd=(intmd or mstage or convlstm))
            
        if self.convlstm:
            output_emb = emb_size
            self.global_out = ConvLSTM(input_size=(32, 32),
                  input_dim=emb_size,
                  hidden_dim=[emb_size],
                  kernel_size=(1, 1),
                  num_layers=1,
                  batch_first=True,
                  bias=True,
                  return_all_layers=False)
            
            self.fcn_out = torch.nn.Sequential(
                  ConvBatchNormReLU(output_emb+8, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                  nn.Conv2d(output_emb//2, 3*5, kernel_size=1))
                  
            self.fcn_out1 = torch.nn.Sequential(
                  ConvBatchNormReLU(2*output_emb+8, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                  nn.Conv2d(output_emb//2, 6*5, kernel_size=1))
        #self.vl_transformer = VisionLanguageEncoder(d_model=512, nhead=8, num_encoder_layers=6,num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False)
        '''
        #transformer
        decoder_layer = TransformerDecoderLayer(512, 8, 2048, 0.1, "relu", False)
        decoder_norm = nn.LayerNorm(512)
        self.decoder = TransformerDecoder(decoder_layer, 6, decoder_norm, return_intermediate=True,d_model=512)
        
        encoder_layer = TransformerEncoderLayer(512, 8, 2048, 0.1, "relu", False)
        encoder_norm = None
        self.encoder = TransformerEncoder(encoder_layer, 6, encoder_norm)
        '''
            
        ## Mapping module
        '''
        for i in self.parameters():
            i.requires_grad=False
        ''' 
        self.mapping_visu2 = ConvBatchNormReLU(512 if self.convlstm else 256+1, emb_size, 3, 1, 1, 1, leaky=leaky)
        self.mapping_visu1 = ConvBatchNormReLU(512+4 if self.convlstm else 256+1, emb_size, 3, 1, 1, 1, leaky=leaky)
        self.mp1 = nn.MaxPool2d(16, stride=16)
        self.mp2 = nn.AvgPool2d(4, stride=4)
        self.mp3 = nn.AvgPool2d(16, stride=16)
        self.mp4 = nn.AvgPool2d(2, stride=2)
        
        self.mapbodyfeature = MLP(512,512,512,2)
        
        self.linecode = MLP(512,128,3,2)
        
        self.poscode = MLP(3,128,512,2)
        
        
        #self.pattention = nn.Conv2d(512,1,1)
        
        #self.l_embed = nn.Embedding(22, 512)
        
        ## output head
        
        #self.maplast = ConvBatchNormReLU(output_emb+8, output_emb, 1, 1, 0, 1, leaky=leaky)
        
        output_emb = emb_size
        if self.mstage:
            self.fcn_out = nn.ModuleDict()
            modules = OrderedDict()
            for n in range(0,NFilm):
                modules["out%d"%n] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb//2, 9*5, kernel_size=1))
            self.fcn_out.update(modules)
        else:
            if self.intmd: 
                output_emb = emb_size*NFilm
            if self.use_sal:
                self.conv1 = nn.Conv2d(1, 2, 4, 4)
                self.conv15 = nn.Conv2d(2, 4, 2, 2)
                self.conv2 = nn.Conv2d(4, 8, 2, 2)
            else:
                self.fcn_out = torch.nn.Sequential(
                        ConvBatchNormReLU(output_emb+8, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                        nn.Conv2d(output_emb//2, 3*5, kernel_size=1))
                self.fcn_out1 = torch.nn.Sequential(
                        ConvBatchNormReLU(2*output_emb+8, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                        nn.Conv2d(output_emb//2, 6*5, kernel_size=1))
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, image, seg, ht, dp, word_id, word_mask):
        ## Visual Module
        batch_size = image.size(0)
        '''
        memory_mask = word_mask.view(batch_size,1,-1,1)
        memory_mask = memory_mask.repeat(1,8,1,1024)
        membed_mask = torch.ones(batch_size, 8, 3, 1024).cuda()
        memory_mask = torch.cat((memory_mask,membed_mask),dim=2)
        memory_mask = word_mask.view(batch_size*8,23,1024)
        print(memory_mask.size())
                
        tgt_key_padding_mask = word_mask
        embed_mask = torch.ones(batch_size,3).cuda()
        tgt_key_padding_mask = torch.cat((tgt_key_padding_mask,embed_mask),dim=1)
        tgt_key_padding_mask = tgt_key_padding_mask.bool()
        '''
        dp = dp.unsqueeze(1)
        seg = seg.unsqueeze(1)
        dp = dp.type(torch.FloatTensor).cuda()
        seg = seg.type(torch.FloatTensor).cuda()

        distxy = distancexy.repeat(batch_size,1,1,1).cuda()
        dist = torch.cat([distxy,dp],dim=1)
        
        seeg = seg.view(batch_size,1,-1)
        seeg = F.normalize(seeg,dim=2,p=1)
        
        dist = dist.view(batch_size,3,-1)
        dist = F.normalize(dist,dim=2,p=1)
        
        #===============================#   
        distfeature = dist.permute(0,2,1)
        #distfeature = distfeature.permute(1,0,2)
        #distfeature = self.posecode(distfeature)
        #distfeature = distfeature.permute(1,0,2)
        
        #===============================#
        bodypositionseg = torch.mul(seeg,dist)
        bodyposition = torch.sum(bodypositionseg,dim=2)
        bodyposition = bodyposition.view(batch_size,1,3)
        
        #bodypfeature = self.poscode(bodyposition)
        #bodypfeature = bodypfeature.view(batch_size,1,-1)
        #bodypfeature = bodyposition.permute(1,0,2)
        #bodypfeature = gen_sineembed_for_position(bodypfeature*512)
        #bodypfeature = bodypfeature.permute(1,0,2)
        #bodypfeature = bodypfeature.view(batch_size,1,-1)
        
        #restdistfeature = distfeature - bodypfeature
        #restdistfeature = F.normalize(restdistfeature,dim=2,p=2)
        
        #restdistfeature = restdistfeature.permute(0,2,1)
        #restdistfeature = restdistfeature.view(batch_size,512,512,512)
        
        #distfeature = distfeature.permute(0,2,1)
        #distfeature = distfeature.view(batch_size,512,512,512)
        
        bodyp = bodyposition.view(batch_size,3,1)
        relatepos = torch.sub(dist,bodyp)
        relatepos = relatepos.view(batch_size,3,-1)
        relatepos = relatepos.permute(0,2,1)
        relateposfeature = self.poscode(relatepos)
        relateposfeature = F.normalize(relateposfeature,dim=2,p=2)
        relateposfeature = relateposfeature.permute(0,2,1)
        relateposfeature = relateposfeature.view(batch_size,512,512,512)
        
        relatepos = F.normalize(relatepos,dim=2,p=2)
        relatepos = relatepos.permute(0,2,1)
        relatepos = relatepos.view(batch_size,3,512,512)
        #restdist = restdist * seg
        
        #====================================================#
        raw_fvisu = self.visumodel(image)
        raw_fvisu = raw_fvisu[1]
        bodyinfo = raw_fvisu
        
        #bodypfeature = bodypfeature.view(batch_size,-1)
        #compute position informations
        ht = ht.type(torch.FloatTensor).cuda()
        ht = ht.view(batch_size,-1,3)
        ht = ht.permute(0,2,1)
        ht = ht.view(batch_size,3,512,512)
        ht = torch.mean(ht,dim=1)
        ht = ht.view(batch_size,1,512,512)
        ht = self.mp1(ht)
                
        rd = self.mp3(relatepos)
        
        bodyinfo = torch.cat((bodyinfo, ht),1)
        bodyinfo = torch.cat((bodyinfo, rd),1)
        
        bodyinfo = self.mapping_visu1(bodyinfo)
        bodyinfo = self.mp2(bodyinfo)
        bodyinfo = self.mapping_visu2(bodyinfo)
        bodyinfo = self.mp2(bodyinfo)
        bodyinfo = self.mp4(bodyinfo)
        
        bodyfeature = bodyinfo.view(batch_size,-1)
        #bodyfeature = torch.cat([bodyinfo,bodypfeature],dim=1)
        #bodypfeature = bodypfeature.view(batch_size,1,-1)
        bodyfeature = self.mapbodyfeature(bodyfeature).view(batch_size,-1)  
        bodyfeature = F.normalize(bodyfeature,dim=1,p=2)
        
        line = self.linecode(bodyfeature)
        line = line.view(batch_size,1,3)
        
        lor = line.view(batch_size,3)
        
        '''
        word_id = []
        word_mask = []
        for uu in range(batch_size):
            if(lor[uu,0]>0):
                word_idt = word_ida[uu,:]
                word_idt = word_idt.unsqueeze(0)
                word_id.append(word_idt)
                
                word_maskt = word_maska[uu,:]
                word_maskt = word_maskt.unsqueeze(0)
                word_mask.append(word_maskt)
            else:
                word_idt = word_idb[uu,:]
                word_idt = word_idt.unsqueeze(0)
                word_id.append(word_idt)
                
                word_maskt = word_maskb[uu,:]
                word_maskt = word_maskt.unsqueeze(0)
                word_mask.append(word_maskt)
                
        word_id = torch.cat(word_id,dim=0).contiguous()
        word_mask = torch.cat(word_mask,dim=0).contiguous()
        '''
        
        relatepos = relatepos.view(batch_size,3,512,512)
        relateposfeature = relateposfeature.view(batch_size,512,512,512)
        
        relatepos = relatepos.view(batch_size,3,-1)
        relateposfeature = relateposfeature.view(batch_size,512,-1)
        line = F.normalize(line,dim=2,p=2)
        #pt3 = torch.matmul(line,restdist)
        
        bodyfeature = bodyfeature.view(batch_size,1,512)
        pt512 = torch.matmul(bodyfeature,relateposfeature)

        #pt3 = pt3.view(batch_size,1,512,512)            
        #attention1 = pt3.view(batch_size,1,512,512)
        attention = pt512.view(batch_size,1,512,512)
        
        #===============================================#
        seg = seg.clamp(max=1)
        seg = -seg+1
        attention = attention * seg
        attention = attention.view(batch_size,1,-1)
        attention = F.softmax(attention,dim=2)
        attention = attention.view(batch_size,1,512,512)
        
        pt = attention
        pt = pt.view(batch_size,1,512,512)
        fvisu = self.mapping_visu(raw_fvisu)
        
        #restdistfeature = restdistfeature.view(batch_size,512,512,512)
        #raw_fvisu = fvisu


























def attt_loss(line,relatepos,attention, bbox, eps=1e-3):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.
    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.
    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """
    loss = 0
    batch = line.size(0)
    bbox = bbox.int()
    for ii in range(batch):
        
        region1 = attention[ii,0,bbox[ii][0]:max(bbox[ii][2],bbox[ii][0]+1),bbox[ii][1]:max(bbox[ii][3],bbox[ii][1]+1)].contiguous()
        region1.view(-1)
        region1 = torch.sum(region1)
        
        relatepos = relatepos.view(batch,3,512,512)
        region2 = relatepos[ii,:,bbox[ii][0]:max(bbox[ii][2],bbox[ii][0]+1),bbox[ii][1]:max(bbox[ii][3],bbox[ii][1]+1)].contiguous()
        region2 = region2.view(3,-1)
        region2 = torch.mean(region2,dim=1)
        region2 = region2.view(3)
        
        region2 = torch.sum(torch.abs(region2-line[ii]))
        #print(region)
        loss += region2+1-region1 #-region1
    loss = loss/batch
    return loss

def depth_loss(input, dp, bbox, gi, gj, best_n_list):
    mseloss = torch.nn.MSELoss(reduction='mean' )
    batch = input.size(0)
    dp = dp.view(batch,-1).float()
    dpmax,_ = torch.max(dp,dim=1)
    dpmax = dpmax.view(batch,-1).float()
    bbox = bbox.int()
    dp = dp/dpmax
    loss = 0
    dp = dp.view(batch,512,512)
    
    for ii in range(batch):
        pred_depth = F.sigmoid(input[ii,best_n_list[ii],-1,gj[ii],gi[ii]])
        target_bbox = dp[ii,bbox[ii][0]:max(bbox[ii][2],bbox[ii][0]+1),bbox[ii][1]:max(bbox[ii][3],bbox[ii][1]+1)].contiguous()
        target_bbox = target_bbox.view(-1)
        target_bbox = torch.mean(target_bbox,dim=0)
        loss += mseloss(pred_depth,target_bbox)
    loss = loss/batch
    loss = loss.float()
    return loss






