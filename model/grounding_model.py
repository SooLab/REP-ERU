from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *
from .convlstm import *
from .modulation import *

import argparse
import collections
import logging
import json
import re
import time
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

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
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = Darknet(config_path='model/yolov3.cfg')
        self.visumodel.load_weights('saved_models/yolov3.weights')
        self.trans = CLIPVisionTransformer(512,8,256)

        self.visumodel4t = Darknetfort(config_path='model/yolov3.cfg')
        self.visumodel4t.load_weights('saved_models/yolov3.weights')

        ## Text model
        self.textmodel = BertModel.from_pretrained(bert_model)
        
        ## Mapping module
        if self.use_paf:
            self.mapping_visu = ConvBatchNormReLU(512+3+1 if self.convlstm else 256+3, emb_size, 1, 1, 0, 1, leaky=leaky)
            self.mp1 = nn.MaxPool2d(16, stride=16)
            self.mp2 = nn.AvgPool2d(16, stride=16)
        else:
            self.mapping_visu = ConvBatchNormReLU(512 if self.convlstm else 256, emb_size, 1, 1, 0, 1, leaky=leaky)

        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.ReLU(),)
        self.mp3 = nn.MaxPool2d(8, stride=8)
        self.mp4 = nn.AvgPool2d(8, stride=8)
        self.mapping_visuf = ConvBatchNormReLU(256 + 4 +1, 256, 1, 1,0, 1, leaky=leaky)
        textdim=emb_size
        self.film = FiLMedConvBlock_multihop(NFilm=NFilm,textdim=textdim,visudim=emb_size,\
            emb_size=emb_size,fusion=fusion,intmd=(intmd or mstage or convlstm))

        ## output head
        output_emb = emb_size
        self.loc_avg = nn.AvgPool2d(16, stride=16)
        self.pt_avg = nn.AvgPool2d(16, stride=16)
        self.ht_avg = nn.AvgPool2d(16, stride=16)
        self.vis_map = ConvBatchNormReLU(512, 128, 3, 1, 1, 1, leaky=leaky)
        self.locationpool = torch.nn.Sequential(
                nn.AvgPool2d(8, stride=8),
                #ConvBatchNormReLU(3, 256, 1, 1, 0, 1, leaky=leaky)
                )
        self.linear1 = torch.nn.Sequential(
                    ConvBatchNormReLU(256,128,8, 8, 0, 1, leaky=leaky),
                    ConvBatchNormReLU(128,32,9, 1, 4, 1, leaky=leaky),
                    nn.MaxPool2d(8, stride=8)
        )
        self.linear2 = nn.Linear(32, 3)
        self.language = nn.Linear(512, 1)
        self.stage0 = torch.nn.Sequential(
                    ConvBatchNormReLU(135, 1024, 1, 1, 0, 1, leaky=leaky)
                )
        self.stage1 = torch.nn.Sequential(
                    ConvBatchNormReLU(1024, 1, 9, 1, 4, 1, leaky=leaky),
                    torch.nn.Upsample(512,mode = 'bilinear' , align_corners = True),
                )
        self.upsample = torch.nn.Upsample(512,mode = 'bilinear' , align_corners = True)
        self.tohyper = torch.nn.Sequential(
                    ConvBatchNormReLU(768, 512, 1, 1, 0, 1, leaky=leaky)
                )
        self.word_projection = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(0.1),
                                             nn.Linear(256, 256),
                                             nn.ReLU())
        #self.tstage0 = torch.nn.Con
        self.center = torch.nn.Sequential(
                    nn.AvgPool2d(16, stride=16)
                )
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
            if self.use_sal:
                self.conv1 = torch.nn.Sequential(
                    nn.AvgPool2d(16)
                )
                self.fcn_out = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb+1, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb//2, 9*5, kernel_size=1))
            else:
                self.fcn_out = torch.nn.Sequential(
                        ConvBatchNormReLU(output_emb, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                        nn.Conv2d(output_emb//2, 9*5, kernel_size=1))
        self.test = Vector(512,16,512)
        self.vectmaxp = torch.nn.Sequential(
                        nn.MaxPool2d(16, stride=16),
                        nn.ReLU()
        )
        self.ptmax = torch.nn.Sequential(
                        #nn.MaxPool2d(8, stride=8),
                        nn.ReLU()
        )
        self.draw = torch.nn.Sequential(
                        nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(256, 1)
    def forward(self, image, dp, ht, word_id, word_mask, gest, bbox, gt, amask, sal,phrase):
        ## Visual Module
        
        batch_size = image.size(0)
        out = self.visumodel(image)
        intemide = self.visumodel(image)[1]
        gest = gest.type(torch.FloatTensor).cuda()
        amask = amask.type(torch.FloatTensor).cuda()
        dp = torch.mul(amask,dp)
        dp = F.normalize(dp.type(torch.FloatTensor).view(batch_size,-1),dim=1,p=float('INF')).view(batch_size,1,512,512).cuda() #* 1.5 
        
        
        raw_fvisu4t = self.visumodel4t(image)
  
        xv, yv = torch.meshgrid([torch.arange(0,512), torch.arange(0,512)]) 
        xv = (xv / 512 ).unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1).cuda()
        yv = (yv / 512 ).unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1).cuda()
        xyz = torch.cat( (xv,yv,dp), dim = 1).cuda()

        gestfraw = torch.mul(gest , amask)
        
        gestf = F.normalize(gestfraw.view(batch_size,1,-1),dim=2,p=1).view(batch_size,1,512,512).repeat(1,3,1,1)
        body = torch.mul(gestf , xyz).view(batch_size, 3, -1)
        body = torch.sum(body,dim=2)

        gtbo = F.normalize(gt.view(batch_size,1,-1),dim=2,p=1).view(batch_size,1,512,512).repeat(1,3,1,1)
        target = torch.mul(gtbo , xyz).view(batch_size, 3, -1)
        target = torch.sum(target,dim=2) - body
        
        xyz_cent = xyz - body.unsqueeze(2).unsqueeze(2).repeat(1,1,512,512)

        t = self.test(torch.cat( (ht.type(torch.FloatTensor).cuda(),xyz_cent) ,dim = 1))
        vloss = 1 - torch.cosine_similarity(t, target, dim=1)
        vectmap = torch.cosine_similarity(xyz_cent , t.unsqueeze(2).unsqueeze(2).repeat(1,1,512,512) , dim = 1).unsqueeze(1) - 0.7

        # cv2.imwrite('output/'+rank+'img.jpg' , imagedraw*255)
        norm = torch.max(gestfraw.reshape(batch_size,-1), dim=1, keepdim = True)[0].detach().unsqueeze(2).unsqueeze(2).repeat(1,1,512,512)
        gestfraw = (gestfraw.unsqueeze(1))/norm
        maxgestvect =  self.ptmax(vectmap  ) #+self.ptmax(gestfraw)
        maxvecter = self.vectmaxp(vectmap ) 
        mid = torch.cat((raw_fvisu4t[2], self.mp3(ht.type(torch.FloatTensor).cuda()), self.mp4(dp.type(torch.FloatTensor).cuda()), self.mp4(vectmap.type(torch.FloatTensor).cuda())),1) 
        #
        mid = self.mapping_visuf(mid)
        raw_fvisu = torch.cat((intemide, self.mp1(ht.type(torch.FloatTensor).cuda()), self.mp2(dp.type(torch.FloatTensor).cuda())),1)
        fvisu = self.mapping_visu(raw_fvisu) * maxvecter.repeat(1,512,1,1).detach()
        raw_fvisu = F.normalize(fvisu, p=2, dim=1)
        size = (raw_fvisu.shape[2])
        
        ## Language Module
        all_encoder_layers, _ = self.textmodel(word_id, \
            token_type_ids=None, attention_mask=word_mask)
        ## Sentence feature at the first position [cls]
        raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
             + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
        raw_fword = (all_encoder_layers[-1] + all_encoder_layers[-2]\
             + all_encoder_layers[-3] + all_encoder_layers[-4])/4
        if not self.tunebert:
            ## fix bert during training
            # raw_flang = raw_flang.detach()
            hidden = raw_flang.detach()
            raw_fword = raw_fword.detach()

        fword = Variable(torch.zeros(raw_fword.shape[0], raw_fword.shape[1], self.emb_size).cuda())
        for ii in range(raw_fword.shape[0]):
            ntoken = (word_mask[ii] != 0).sum()
            fword[ii,:ntoken,:] = F.normalize(self.mapping_lang(raw_fword[ii,:ntoken,:]), p=2, dim=1)
            ## [CLS], [SEP]
            # fword[ii,1:ntoken-1,:] = F.normalize(self.mapping_lang(raw_fword[ii,1:ntoken-1,:].view(-1,self.textdim)), p=2, dim=1)
        raw_fword = fword
        x = self.trans(mid)[1].reshape(batch_size,256,-1).permute(0,2,1)
        x = self.linear(x)
        pt = self.upsample ( torch.softmax(x , dim = 1).squeeze(2).reshape(batch_size,1,64,64) )

        gt = gt.unsqueeze(1)
        gest = 1 - torch.mul(gest , amask).clamp(max = 1,min=0)
        pt = torch.mul(pt, amask.unsqueeze(1))
        pt = F.normalize(pt.view(batch_size,1,-1),dim=2,p=1).view(batch_size,1,512,512)
        loss3  = 1 - torch.sum(torch.mul(pt,gt).reshape(batch_size,-1) , -1)
        gt = F.normalize(gt.view(batch_size,1,-1),dim=2,p=1).view(batch_size,1,512,512)
        eps = 1e-7

        vect = torch.mul(pt , xyz).view(batch_size, 3, -1)
        vect = torch.sum(vect,dim=2) - body
        loss1 = torch.sum(torch.abs(vect - target))
        loss3  += torch.sum( (torch.log( gt / (eps + pt) + eps ) * gt).reshape(batch_size,-1) , -1) * 0.1

        norm = torch.max(pt.reshape(batch_size,-1), dim=1, keepdim = True)[0].detach().unsqueeze(2).unsqueeze(2).repeat(1,1,512,512)
        pt = (pt/norm).detach() 
        centerout = self.center(pt.type(torch.FloatTensor)).squeeze(1).cuda()

        coord = generate_coord(batch_size, raw_fvisu.size(2), raw_fvisu.size(3))
        x, attnscore_list = self.film(raw_fvisu, raw_fword, coord,maxvecter,fsent=None,word_mask=word_mask)
        if self.mstage:
            outbox = []
            for film_ii in range(len(x)):
                outbox.append(self.fcn_out["out%d"%film_ii](x[film_ii]))
        elif self.convlstm:
            x = torch.stack(x, dim=1)

            output, state = self.global_out(x)
            output, hidden, cell = output[-1], state[-1][0], state[-1][1]
            if self.use_sal:
                #pt = sal.type(torch.FloatTensor).cuda()
                pt_c = self.conv1(pt.type(torch.FloatTensor).cuda())

                hidden = torch.cat((hidden, pt_c), 1)

            outbox = [self.fcn_out(hidden)]
        else:
            x = torch.stack(x, dim=1).view(batch_size, -1, raw_fvisu.size(2), raw_fvisu.size(3))
            outbox = [self.fcn_out(x)]

        return outbox, attnscore_list, loss1, vloss, centerout,loss3,pt   ## list of (B,N,H,W)


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.append('.')
    from dataset.data_loader import *
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from utils.transforms import ResizeImage, ResizeAnnotation
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--size', default=416, type=int,
                        help='image size')
    parser.add_argument('--data', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--split', default='train', type=str,
                        help='name of the dataset split used to train')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=256, type=int,
                        help='word embedding dimensions')
    # parser.add_argument('--lang_layers', default=3, type=int,
    #                     help='number of SRU/LSTM stacked layers')

    args = parser.parse_args()

    torch.manual_seed(13)
    np.random.seed(13)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    input_transform = Compose([
        ToTensor(),
        # ResizeImage(args.size),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    refer = ReferDataset(data_root=args.data,
                         dataset=args.dataset,
                         split=args.split,
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)

    train_loader = DataLoader(refer, batch_size=1, shuffle=True,
                              pin_memory=True, num_workers=0)

#    model = textcam_yolo_light(emb_size=args.emb_size)
    
    for i in enumerate(train_loader):
        print(i)
        break
