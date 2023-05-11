###############################################################
##### @Title:  UWMGI infer trick
##### @Time:  2022/6/9
##### @Author: frank
##### @Describe: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
###############################################################
import os
import pdb
from tkinter.messagebox import NO
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
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html

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
        height = heights[idx].item()
        width = widths[idx].item()
        msk = cv2.resize(msks[idx], 
                        dsize=(width, height), 
                        interpolation=cv2.INTER_NEAREST) # back to original shape
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
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            A.OneOf([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
        }
    return data_transforms

class build_dataset(Dataset):
    def __init__(self, df, label=True, transforms=None, cfg=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist() # image
        self.ids = df['id'].tolist()

        if 'mask_path' in df.columns:
            self.mask_paths  = df['mask_path'].tolist() # mask
        else:
            self.mask_paths = None

        self.transforms = transforms
        self.n_25d_shift = cfg.n_25d_shift

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        #### load id
        id       = self.ids[index]
        #### load image
        img_path  = self.img_paths[index]
        img = self.load_2_5d_slice(img_path) # [h, w, c]
        h, w = img.shape[:2]
        
        if self.label: # train
            #### load mask
            mask_path = self.mask_paths[index]
            mask = np.load(mask_path).astype('float32')
            mask/=255.0 # scale mask to [0, 1]

            ### augmentations
            data = self.transforms(image=img, mask=mask)
            img  = data['image']
            mask  = data['mask']
            img = np.transpose(img, (2, 0, 1)) # [h, w, c] => [c, h, w]
            mask = np.transpose(mask, (2, 0, 1)) # [h, w, c] => [c, h, w]
            return torch.tensor(img), torch.tensor(mask)
        
        else:  # test
            ### augmentations
            data = self.transforms(image=img)
            img  = data['image']
            img = np.transpose(img, (2, 0, 1)) # [h, w, c] => [c, h, w]
            return torch.tensor(img), id, h, w

    ###############################################################
    ##### >>>>>>> trick: construct 2.5d slice images <<<<<<
    ###############################################################
    def load_2_5d_slice(self, middle_img_path):
        #### 步骤1: 获取中间图片的基本信息
        #### eg: middle_img_path: 'slice_0005_266_266_1.50_1.50.png' 
        middle_slice_num = os.path.basename(middle_img_path).split('_')[1] # eg: 0005
        middle_str = 'slice_'+middle_slice_num
        middle_img = cv2.imread(middle_img_path, cv2.IMREAD_UNCHANGED)

        new_25d_imgs = []

        ##### 步骤2：按照左右n_25d_shift数量进行填充，如果没有相应图片填充为Nan.
        ##### 注：经过EDA发现同一天的所有患者图片的shape是一致的
        for i in range(-self.n_25d_shift, self.n_25d_shift+1): # eg: i = {-2, -1, 0, 1, 2}
            shift_slice_num = int(middle_slice_num) + i
            shift_str = 'slice_'+str(shift_slice_num).zfill(4)
            shift_img_path = middle_img_path.replace(middle_str, shift_str)
            
            if os.path.exists(shift_img_path):
                shift_img = cv2.imread(shift_img_path, cv2.IMREAD_UNCHANGED) # [w, h]
                new_25d_imgs.append(shift_img)
            else:
                new_25d_imgs.append(None)
        
        ##### 步骤3：从中心开始往外循环，依次填补None的值
        ##### eg: n_25d_shift = 2, 那么形成5个channel, idx为[0, 1, 2, 3, 4], 所以依次处理的idx为[1, 3, 0, 4]
        shift_left_idxs = []
        shift_right_idxs = []
        for related_idx in range(self.n_25d_shift):
            shift_left_idxs.append(self.n_25d_shift - related_idx - 1)
            shift_right_idxs.append(self.n_25d_shift + related_idx + 1)

        for left_idx, right_idx in zip(shift_left_idxs, shift_right_idxs):
            if new_25d_imgs[left_idx] is None:
                new_25d_imgs[left_idx] = new_25d_imgs[left_idx+1]
            if new_25d_imgs[right_idx] is None:
                new_25d_imgs[right_idx] = new_25d_imgs[right_idx-1]

        new_25d_imgs = np.stack(new_25d_imgs, axis=2).astype('float32') # [w, h, c]
        mx_pixel = new_25d_imgs.max()
        if mx_pixel != 0:
            new_25d_imgs /= mx_pixel
        return new_25d_imgs

def build_dataloader(df, fold, data_transforms, CFG):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = build_dataset(train_df, label=True, transforms=data_transforms['train'], cfg=CFG)
    valid_dataset = build_dataset(valid_df, label=True, transforms=data_transforms['valid_test'], cfg=CFG)

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=CFG.num_worker, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=CFG.num_worker, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
def build_model(CFG, test_flag=False):
    if test_flag:
        pretrain_weights = None
    else:
        pretrain_weights = CFG.pretrain_weights
    model = smp.Unet(
            encoder_name=CFG.backbone,
            encoder_weights=pretrain_weights, 
            in_channels=2*CFG.n_25d_shift+1,             
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
        
            bce_loss = losses_dict["BCELoss"](y_preds, masks)
            tverskly_loss = losses_dict["TverskyLoss"](y_preds, masks)
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
def test_one_epoch(ckpt_paths_dict, test_loader, CFG):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids, h, w) in pbar:

        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        size = images.size()
        masks = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32) # [b, c, w, h]
        
        ############################################
        ##### >>>>>>> cross validation & ensemble infer <<<<<<
        ############################################
        total_ckpt_paths = len(ckpt_paths_dict) * CFG.n_fold

        for backbone_name, ckpt_paths in ckpt_paths_dict.items():
            CFG.backbone = backbone_name
            for sub_ckpt_path in ckpt_paths:
                model = build_model(CFG, test_flag=True)
                model.load_state_dict(torch.load(sub_ckpt_path))
                model.eval()
                y_preds = model(images) # [b, c, w, h]
                y_preds   = torch.nn.Sigmoid()(y_preds)
                masks += y_preds

                #x,y,xy flips as TTA
                if CFG.tta:
                    flips = [[-1],[-2],[-2,-1]]
                    for f in flips:
                        images_f = torch.flip(images, f)
                        y_preds = model(images_f) # [b, c, w, h]
                        y_preds = torch.flip(y_preds, f)
                        y_preds   = torch.nn.Sigmoid()(y_preds)
                        masks += y_preds

        if CFG.tta:
            total_ckpt_paths = len(ckpt_paths_dict) * CFG.n_fold * 4
        else:
            total_ckpt_paths = len(ckpt_paths_dict) * CFG.n_fold
        masks /= total_ckpt_paths
        
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
        seed = 42  # birthday
        num_worker = 16 # debug => 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "ckpt-frank"

        # step2: data
        n_25d_shift = 1
        n_fold = 4
        img_size = [224, 224]
        train_bs = 128
        valid_bs = train_bs * 2

        # step3: model
        backbone_1 = 'resnext50_32x4d'
        ckpt_name_1 = "resnext50_32x4d_img224224_bs128_fold4_2.5d_shift1"
        backbone_2 = 'resnext101_32x8d'
        ckpt_name_2 = "resnext101_32x8d_img224224_bs128_fold4_2.5d_shift1"

        pretrain_weights = "imagenet"
        num_classes = 3

        # step4: optimizer
        epoch = 12
        lr = 1e-3
        wd = 1e-5
        lr_drop = 8

        # step5: infer
        thr = 0.45
        tta = True
    
    test_flag = True
    if test_flag:
        set_seed(CFG.seed)
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
        if not len(sub_df):
            sub_firset = True
            sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000*3]
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

        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(test_df, label=False, transforms=data_transforms['valid_test'], cfg=CFG)
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=2, shuffle=False, pin_memory=False)

        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        ckpt_path_1 = f"../input/{CFG.ckpt_fold}/{CFG.ckpt_name_1}"
        ckpt_path_2 = f"../input/{CFG.ckpt_fold}/{CFG.ckpt_name_2}"
        ckpt_paths_1  = glob(f'{ckpt_path_1}/best*')
        ckpt_paths_2  = glob(f'{ckpt_path_2}/best*')
        assert len(ckpt_paths_1) == CFG.n_fold and len(ckpt_paths_2) == CFG.n_fold, "ckpt path error!"

        ckpt_paths_dict = {CFG.backbone_1:ckpt_paths_1, CFG.backbone_2:ckpt_paths_2}
        pred_strings, pred_ids, pred_classes = test_one_epoch(ckpt_paths_dict, test_loader, CFG)

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