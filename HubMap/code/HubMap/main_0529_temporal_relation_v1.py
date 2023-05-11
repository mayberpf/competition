###############################################################
##### @Title:  UWMGI baseline
##### @Time:  2022/5/29
##### @Author: frank
##### @Describe: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @To do: 
        #  data: multiresolution...
        #  model: resnxt, swin..
        #  loss: lovasz, HaussdorfLoss..
        #  infer: tta & pseudo labels...
##### @Reference:
        # UWMGI: Unet [Train] [PyTorch]: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/
        # UWMGI: Unet [Infer] [PyTorch]: https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch/
###############################################################
import os
import pdb
from tty import CFLAG
import cv2
import time
import glob
import random

from cv2 import transform
import cupy as cp # https://cupy.dev/ => pip install cupy-cuda102
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from sklearn.model_selection import StratifiedGroupKFold # Sklearn
import albumentations as A # Augmentations
import segmentation_models_pytorch as smp # smp

def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################
##### part0: data preprocess
###############################################################
def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    # row['id'] = f'case{case}_day{day}_slice_{slice_}'
    return row

def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = cp.array(msk)
    pixels = msk.flatten()
    pad    = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs   = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        msk = msks[idx]
        height = heights[idx].item()
        width = widths[idx].item()
        shape0 = np.array([height, width])
        resize = np.array([320, 384])
        if np.any(shape0!=resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0//2, pad0//2 + pad0%2]
            padx = [pad1//2, pad1//2 + pad1%2]
            msk = msk[pady[0]:-pady[1], padx[0]:-padx[1], :]
            msk = msk.reshape((*shape0, 3))
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes

###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
# document: https://albumentations.ai/docs/
# example: https://github.com/albumentations-team/albumentations_examples
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            # dimension should be multiples of 32.
            A.OneOf([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1.0),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),

            ], p=1.0),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
        }
    return data_transforms


def load_test_img(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape0 = np.array(img.shape[:2])
    resize = np.array(size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        img = np.pad(img, [pady, padx])
        img = img.reshape((*resize))
    return img, shape0

def load_test_imgs(img_paths, size):
    imgs = np.zeros((*size, len(img_paths)), dtype=np.float32)
    for i, img_path in enumerate(img_paths):
        if i==0:
            img, shape0 = load_test_img(img_path, size=size)
        else:
            img, _ = load_test_img(img_path, size=size)
        img = img.astype('float32') # original is uint16
        mx = np.max(img)
        if mx:
            img/=mx # scale image to [0, 1]
        imgs[..., i]+=img
    return imgs, shape0

class build_dataset(Dataset):
    def __init__(self, df, label=True, transforms=None, CFG=None):
        self.df = df
        self.label = label
        self.ids        = df['id'].tolist()
        self.cfg = CFG

        if 'mask_path' in df.columns:
            self.img_paths  = df['image_path'].tolist()
            self.mask_paths  = df['mask_path'].tolist()
        else:
            self.img_paths  = df['image_paths'].tolist()

        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        
        if self.label:
            
            # load img - 已经处理好了
            img = np.load(img_path) # [h, w, c]
            img = img.astype('float32') # original is uint16
            mx = np.max(img)
            if mx:
                img/=mx # scale image to [0, 1]
         
            # load mask - 已经处理好了
            mask_path = self.mask_paths[index]
            msk = np.load(mask_path)
            msk = msk.astype('float32')
            msk/=255.0

            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
        
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
    
        else:
            id_ = self.ids[index]
            img, shape0 = load_test_imgs(img_path, self.cfg.img_size)
            h, w = shape0
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), id_, h, w


def build_dataloader(df, fold, data_transforms, CFG):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = build_dataset(train_df, label=True, transforms=data_transforms['train'], CFG=CFG)
    valid_dataset = build_dataset(valid_df, label=True, transforms=data_transforms['valid_test'], CFG=CFG)

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://smp.readthedocs.io/en/latest/encoders_timm.html
def build_model(CFG, test_flag=False):
    if test_flag:
        pretrain_weights = None
    else:
        pretrain_weights = "imagenet"
    model = smp.Unet(
            encoder_name=CFG.backbone,
            encoder_weights=pretrain_weights, 
            in_channels=3,             
            classes=CFG.num_classes,   
            activation=None,
        )
    model.to(CFG.device)
    return model

###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
def build_loss():
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
    return {"BCELoss":BCELoss, "TverskyLoss":TverskyLoss}

###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    # dice: A ^ B / A U B
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou
    
###############################################################
##### >>>>>>> part5: train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, bce_all, tverskly_all = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, masks) in pbar:
        optimizer.zero_grad()

        images = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks  = masks.to(CFG.device, dtype=torch.float)  # [b, c, w, h]

        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]
        
            bce_loss = 0.5 * losses_dict["BCELoss"](y_preds, masks)
            tverskly_loss = 0.5 * losses_dict["TverskyLoss"](y_preds, masks)
            losses = bce_loss + tverskly_loss
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item() / images.shape[0]
        bce_all += bce_loss.item() / images.shape[0]
        tverskly_all += tverskly_loss.item() / images.shape[0]
    
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, bce_all: {:.3f}, tverskly_all: {:.3f}".format(losses_all, bce_all, tverskly_all), flush=True)
        
@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    val_scores = []
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, masks) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks   = masks.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        
        y_preds = model(images) 
        y_preds   = torch.nn.Sigmoid()(y_preds) # [b, c, w, h]
        
        val_dice = dice_coef(masks, y_preds).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_preds).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
    val_scores  = np.mean(val_scores, axis=0)
    val_dice, val_jaccard = val_scores
    print("val_dice: {:.4f}, val_jaccard: {:.4f}".format(val_dice, val_jaccard), flush=True)
    
    return val_dice, val_jaccard

@torch.no_grad()
def test_one_epoch(ckpt_paths, test_loader, CFG):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids, h, w) in pbar:
        
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        size = images.size()
        masks = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32) # [b, c, w, h]
        
        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################
        for sub_ckpt_path in ckpt_paths:
            model = build_model(CFG, test_flag=True)
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            y_preds = model(images) # [b, c, w, h]
            y_preds   = torch.nn.Sigmoid()(y_preds)
            masks += y_preds/len(ckpt_paths)
        
        masks = (masks.permute((0, 2, 3, 1))>CFG.thr).to(torch.uint8).cpu().detach().numpy() # [n, h, w, c]
        result = masks2rles(masks, ids, h, w)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
    return pred_strings, pred_ids, pred_classes


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "ckpt-frank"
        ckpt_name = "efficientnetb1_img224224_bs128_fold4_2.5d"  # for submit. # 
        # step2: data
        n_fold = 5
        img_size = [320, 384]
        train_bs = 64
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'efficientnet-b0'
        num_classes = 3
        # step4: optimizer
        epoch = 15
        lr = 1e-3
        wd = 1e-5
        lr_drop = 10
        # step5: infer
        thr = 0.4 # hyper-parameter
    
    set_seed(CFG.seed)
    ckpt_path = f"../input/{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = True
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        path_df = pd.DataFrame(glob('../input/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])

        path_df['mask_path'] = path_df.image_path.str.replace('image','mask')

        path_df['image_path'] = path_df.image_path.str.replace('/kaggle/','../') # .str: 特定列应用python字符串处理方法 
        path_df['mask_path'] = path_df.mask_path.str.replace('/kaggle/','../')
        pdb.set_trace()
        path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy',''))

        df = pd.read_csv('../input/uwmgi-mask-dataset/train.csv')
        df['segmentation'] = df.segmentation.fillna('')
        df['rle_len'] = df.segmentation.map(len) # length of each rle mask

        df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
        df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

        df = df.drop(columns=['segmentation', 'class', 'rle_len'])
        df = df.groupby(['id']).head(1).reset_index(drop=True)
        df = df.merge(df2, on=['id'])
        df['empty'] = (df.rle_len==0) # empty masks

        df = df.drop(columns=['image_path','mask_path'])
        df = df.merge(path_df, on=['id'])

        fault1 = 'case7_day0'
        fault2 = 'case81_day30'
        df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)

        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################

        skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
            df.loc[val_idx, 'fold'] = fold
        
        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ##### build_transforme() & build_dataset() & build_dataloader()
            ##### build_model() & build_loss()
            ###############################################################
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms, CFG) # dataset & dtaloader
            model = build_model(CFG) # model
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss

            best_val_dice = 0
            best_epoch = 0
            
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_dice, val_jaccard = valid_one_epoch(model, valid_loader, CFG)
                
                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = (val_dice > best_val_dice)
                best_val_dice = max(best_val_dice, val_dice)
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{}\n".format(epoch, epoch_time, best_val_dice), flush=True)


    test_flag = False
    if test_flag:
        set_seed(CFG.seed)
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        # "sample_submission.csv" ==> when you click submit, new one will be swaped and populate the text directory.
        # ref: https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-basic-submission/notebook 

        sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
        if not len(sub_df):
            sub_firset = True
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
            sub_df = sub_df[~sub_df.segmentation.isna()][:1000*3]
            sub_df = sub_df.drop(columns=['class','segmentation']).drop_duplicates()
            paths = glob(f'../input/uw-madison-gi-tract-image-segmentation/train/**/*png',recursive=True)
        else:
            sub_firset = False
            sub_df = sub_df.drop(columns=['class','predicted']).drop_duplicates()
            paths = glob(f'../input/uw-madison-gi-tract-image-segmentation/test/**/*png',recursive=True)
        sub_df = sub_df.apply(get_metadata,axis=1)
        path_df = pd.DataFrame(paths, columns=['image_path'])
        path_df = path_df.apply(path2info, axis=1)
        test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')

        # 2.5D MetaData
        channels=3
        stride=2
        for i in range(channels):
            test_df[f'image_path_{i:02}'] = test_df.groupby(['case','day'])['image_path'].shift(-i*stride).fillna(method="ffill")
        test_df['image_paths'] = test_df[[f'image_path_{i:02d}' for i in range(channels)]].values.tolist()
        if sub_firset:
            test_df = test_df.sample(frac=1.0)
        # test_df.image_paths[0]

        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(test_df, label=False, transforms=data_transforms['valid_test'], CFG=CFG)
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=2, shuffle=False, pin_memory=False)
    
        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        # attention: change the corresponding upload path to kaggle.
        ckpt_paths  = glob(f'{ckpt_path}/best*')
        assert len(ckpt_paths) == CFG.n_fold, "ckpt path error!"

        pred_strings, pred_ids, pred_classes = test_one_epoch(ckpt_paths, test_loader, CFG)

        ###############################################################
        ##### step3: submit
        ###############################################################
        pred_df = pd.DataFrame({
            "id":pred_ids,
            "class":pred_classes,
            "predicted":pred_strings
        })
        if not sub_firset:
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
            del sub_df['predicted']
        else:
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000*3]
            del sub_df['segmentation']
            
        sub_df = sub_df.merge(pred_df, on=['id','class'])
        sub_df.to_csv('submission.csv',index=False)

        