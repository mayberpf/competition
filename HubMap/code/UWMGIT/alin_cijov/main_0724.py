###############################################################
##### @Title:  HuBMAP baseline
##### @Time:  2022/07/24
##### @Author: frank
##### @Describe: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @Reference:
        # https://www.kaggle.com/code/vexxingbanana/hubmap-unet-semantic-approach-train
        # https://www.kaggle.com/code/vexxingbanana/hubmap-unet-semantic-approach-infer
###############################################################
import os
import pdb
import cv2
import time
import glob
import random
import tifffile

from cv2 import transform
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html

from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
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
#Modified from https://www.kaggle.com/code/abhinand05/hubmap-extensive-eda-what-are-we-hacking
def read_tiff(path, scale=None, verbose=0): 
    image = tifffile.imread(path)
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    
    if verbose:
        print(f"[{path}] Image shape: {image.shape}")
    
    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        if verbose:
            print(f"[{path}] Resized Image shape: {image.shape}")
        
    mx = np.max(image)
    image = image.astype(np.float32)
    if mx:
        image /= mx # scale image to [0, 1]
    return image

# Ref: https://www.kaggle.com/code/paulorzp/rle-functions-run-lenght-encode-decode/script
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            A.OneOf([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),
            A.HorizontalFlip(p=0.5),
            ], p=1.0),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
        }
    return data_transforms

class build_dataset(Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        img_path = self.df.loc[index, 'image_path']
        img_height = self.df.loc[index, 'img_height']
        img_width = self.df.loc[index, 'img_width']
        id_ = self.df.loc[index, 'id']
        img = read_tiff(img_path)
        
        if self.label:
            rle_mask = self.df.loc[index, 'rle']
            mask = rle_decode(rle_mask, (img_height, img_width))
            pdb.set_trace() 
            if self.transforms:
                data = self.transforms(image=img, mask=mask)
                img  = data['image']
                mask  = data['mask']
            
            mask = np.expand_dims(mask, axis=0)
            img = np.transpose(img, (2, 0, 1))
            # mask = np.transpose(mask, (2, 0, 1))
            
            return torch.tensor(img), torch.tensor(mask)
        
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
                
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), img_height, img_width, id_
        

def build_dataloader(df, fold, data_transforms, CFG):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = build_dataset(train_df, label=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_df, label=True, transforms=data_transforms['valid_test'])
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=CFG.num_worker, 
                              shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=CFG.num_worker, 
                              shuffle=False, pin_memory=True)
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
    DiceLoss    = smp.losses.DiceLoss(mode='binary')
    return {"BCELoss":BCELoss, "DiceLoss":DiceLoss}

###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
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
    losses_all, bce_all, dice_all = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, masks) in pbar:
        optimizer.zero_grad()

        images = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks  = masks.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        
        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]

            bce_loss = 0.5 * losses_dict["BCELoss"](y_preds, masks)
            dice_loss = 0.5 * losses_dict["DiceLoss"](y_preds, masks)
            losses = bce_loss + dice_loss
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item() / images.shape[0]
        bce_all += bce_loss.item() / images.shape[0]
        dice_all += dice_loss.item() / images.shape[0]
    
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, bce_all: {:.3f}, dice_all: {:.3f}".format(losses_all, bce_all, dice_all), flush=True)
        
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
    pred_ids = []
    pred_rles = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, heights, widths, ids) in pbar:

        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        size = images.size()
        masks = torch.zeros((size[0], CFG.num_classes, size[2], size[3]), device=CFG.device, dtype=torch.float32) # [b, c, w, h]
        
        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################
        for sub_ckpt_path in ckpt_paths:
            model = build_model(CFG, test_flag=True)
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            y_preds = model(images) # [b, c, w, h]
            y_preds = torch.nn.Sigmoid()(y_preds)
            masks += y_preds/len(ckpt_paths)
        
        masks = (masks.permute((0, 2, 3, 1))>CFG.thr).to(torch.uint8).cpu().detach().numpy() # [n, h, w, c]
 
        for idx in range(masks.shape[0]):
            height = heights[idx].item()
            width = widths[idx].item()
            id_ = ids[idx].item()
            msk = cv2.resize(masks[idx].squeeze(), dsize=(width, height), interpolation=cv2.INTER_NEAREST)
            rle = rle_encode(msk)
            pred_rles.append(rle)
            pred_ids.append(id_)
    
    return pred_ids, pred_rles


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_worker = 0 # if debug
        data_path = "../input/hubmap-organ-segmentation"
        ckpt_path = "../input/ckpt-frank/resnet18_img512512_bs16_fold2" # for submit
        # step2: data
        n_fold = 2
        img_size = [512, 512]
        train_bs = 32
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'resnet18'
        num_classes = 1
        # step4: optimizer
        epoch = 12
        lr = 1e-3
        wd = 1e-5
        lr_drop = 8
        # step5: infer
        thr = 0.5
    
    set_seed(CFG.seed)
    if not os.path.exists(CFG.ckpt_path):
        os.makedirs(CFG.ckpt_path)

    train_val_flag = True
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        df = pd.read_csv(os.path.join(CFG.data_path, "train.csv"))

        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold
        df['image_path'] = df['id'].apply(lambda x: os.path.join(CFG.data_path, 'train_images', str(x) + '.tiff'))
     
        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ###############################################################
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms, CFG) # dataset & dtaloader
            
            model = build_model(CFG) # model
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd) # optimizer
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
                    save_path = f"{CFG.ckpt_path}/best_fold{fold}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_dice), flush=True)


    test_flag = False
    if test_flag:
        set_seed(CFG.seed)
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        df = pd.read_csv(os.path.join(CFG.data_path, "test.csv"))
        df['image_path'] = df['id'].apply(lambda x: os.path.join(CFG.data_path, 'test_images', str(x) + '.tiff'))

        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(df, label=False, transforms=data_transforms['valid_test'])
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=2, shuffle=False, pin_memory=False)

        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        # attention: change the corresponding upload path to kaggle.
        ckpt_paths  = glob(f'{CFG.ckpt_path}/best*')
        assert len(ckpt_paths) == CFG.n_fold, "ckpt path error!"
        pred_ids, pred_rles = test_one_epoch(ckpt_paths, test_loader, CFG)

        ###############################################################
        ##### step3: submit
        ###############################################################
        pred_df = pd.DataFrame({
            "id":pred_ids,
            "rle":pred_rles
        })
        pred_df.to_csv('submission.csv',index=False)

        