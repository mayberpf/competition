###############################################################
##### @Title:  HuBMAP baseline
##### @Time:  2022/07/31
##### @Author: frank
##### @Struct: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @Describe: 
        # 青蛙哥的代码. 但是先简单的把模型替换为resnet
##### @Reference:
        # LB 0.75 Variable size swin transformer v1 and v2: https://www.kaggle.com/code/hengck23/lb-0-75-variable-size-swin-transformer-v1-and-v2 
        # hubmap-discuss-00: https://www.kaggle.com/datasets/hengck23/hubmap-discuss-00
        # [Training] HuBMAP LB 0.75 Swin Transformer v1: https://www.kaggle.com/code/alincijov/training-hubmap-lb-0-75-swin-transformer-v1 
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
from utils import *
from models import *
from fastai.vision.all import *

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

def train_augment(image, mask):
    image, mask = do_random_flip(image, mask)
    image, mask = do_random_rot90(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_noise(image, mask, mag=0.1),
        lambda image, mask: do_random_contast(image, mask, mag=0.40),
        lambda image, mask: do_random_hsv(image, mask, mag=[0.40, 0.40, 0])
    ], 2): image, mask = fn(image, mask)

    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_rotate_scale(image, mask, angle=45, scale=[0.50, 2.0]),
    ], 1): image, mask = fn(image, mask)

    return image, mask


def valid_augment(image, mask):
    #image, mask  = do_crop(image, mask, image_size, xy=(None,None))
    return image, mask


class build_dataset(Dataset):
    def __init__(self, df, label=True, transforms=None, CFG=None):
        ###########################################
        ##### >>>>>>> Use "hubmap-2022-256x256" Dataset <<<<<<
        ############################################
        self.df = df
        ids = df.id.astype(str).values
        self.third_data_train = os.path.join(CFG.third_data_path, "train")
        self.third_data_mask = os.path.join(CFG.third_data_path, "masks")
        self.file_names = [file_name for file_name in os.listdir(self.third_data_train) if file_name.split('_')[0] in ids]
        self.organ_to_label = {'kidney' : 0, 'prostate' : 1, 'largeintestine' : 2, 'spleen' : 3, 'lung' : 4}        
        self.label = label
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        if self.label:
            file_name = self.file_names[index]
            sub_df = self.df.iloc[index]
            organ = self.organ_to_label[sub_df.organ]
            
            image = cv2.cvtColor(cv2.imread(os.path.join(self.third_data_train, file_name)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
            mask = cv2.imread(os.path.join(self.third_data_mask, file_name), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            image = cv2.resize(image,dsize=(CFG.img_size, CFG.img_size),interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask, dsize=(CFG.img_size, CFG.img_size),interpolation=cv2.INTER_LINEAR)
            
            if self.transforms:
                image, mask = self.transforms(image, mask)
            
            data_info ={}
            data_info['index']= index
            data_info['id'] = file_name
            data_info['organ'] = torch.tensor([organ], dtype=torch.long)
            image = image[:,:,::-1].transpose(2,0,1) # BGR => RGB
            image = np.ascontiguousarray(image)
            data_info['image'] = torch.tensor(image, dtype=torch.float)
            data_info['mask' ] = torch.tensor(mask>0.5, dtype=torch.float)
            return data_info
        
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
                
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), img_height, img_width, id_
        

def build_dataloader(df, fold, data_transforms, CFG):
    
    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)
    
    train_dataset = build_dataset(train_df, label=True, transforms=data_transforms['train'], CFG=CFG)
    valid_dataset = build_dataset(valid_df, label=True, transforms=data_transforms['valid_test'], CFG=CFG)
    
    def null_collate(batch):
        d = {}
        key = batch[0].keys()
        for k in key:
            v = [b[k] for b in batch]
            if k in ['mask', 'image', 'organ']:
                v = torch.stack(v)
            d[k] = v

        d['mask'] = d['mask'].unsqueeze(1)
        d['organ'] = d['organ'].reshape(-1)
        return d
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=CFG.num_worker, 
                              shuffle=True, pin_memory=True, drop_last=False, collate_fn = null_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=CFG.num_worker, 
                              shuffle=False, pin_memory=True, collate_fn = null_collate)
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://smp.readthedocs.io/en/latest/encoders_timm.html
def build_model(CFG, test_flag=False):
    model = Net()
    
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
def compute_dice_score(probability, mask):
    N = len(probability)
    p = probability.reshape(N,-1)
    t = mask.reshape(N,-1)

    p = p>0.5
    t = t>0.5
    uion = p.sum(-1) + t.sum(-1)
    overlap = (p*t).sum(-1)
    dice = 2*overlap/(uion+0.0001)
    return dice

def dice_coef(y_pred, y_true,  thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


###############################################################
##### >>>>>>> part5: train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, bce_all, dice_all = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, batch in pbar:
        # batch: dict_keys(['index', 'id', 'organ', 'image', 'mask'])
        optimizer.zero_grad()

        images = batch['image'].to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks  = batch['mask'].to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        
        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]
            
            bce_loss = losses_dict["BCELoss"](y_preds, masks)
            # dice_loss = losses_dict["DiceLoss"](y_preds, masks)
            losses = bce_loss # + dice_loss
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item() / images.shape[0]
        bce_all += bce_loss.item() / images.shape[0]
        # dice_all += dice_loss.item() / images.shape[0]
        dice_all += 0
    
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, bce_all: {:.3f}, dice_all: {:.3f}".format(losses_all, bce_all, dice_all), flush=True)
    
        
@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    
    valid_image = []
    valid_probability = []
    valid_mask = []
    valid_id = []
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, batch in pbar:
        images = batch['image'].to(CFG.device, dtype=torch.float) # [b, c, w, h]
        masks = batch['mask'].to(CFG.device, dtype=torch.float) # [b, c, w, h]
        
        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]
            y_preds   = torch.nn.Sigmoid()(y_preds) # [b, c, w, h]
        
        valid_image.append(images)
        valid_probability.append(y_preds)
        valid_mask.append(masks)
        valid_id.extend(batch['id'])
        
    images = torch.cat(valid_image)
    probabilitys = torch.cat(valid_probability)
    masks = torch.cat(valid_mask)
    val_dice = dice_coef(probabilitys, masks)
        
    print("val_dice: {:.4f}".format(val_dice), flush=True)
    
    return val_dice, images, probabilitys, masks, valid_id


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



def plot_visual(image, mask, pred, image_id, cmap):
    plt.figure(figsize=(16, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(image.transpose(1, 2, 0))
    plt.grid(visible=False)
    plt.title("image", fontsize=10)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image.transpose(1, 2, 0))
    plt.imshow(mask.transpose(1, 2, 0), cmap=cmap, alpha=0.5)
    plt.title(f"mask", fontsize=10)    
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(image.transpose(1, 2, 0))
    plt.imshow(pred.transpose(1, 2, 0), cmap=cmap, alpha=0.5)
    plt.title(f"pred", fontsize=10)    
    plt.axis("off")

    plt.savefig(f"./result/{image_id}")
    # plt.show()
    

if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_worker = 0 # 0 if debug. 16 if train by "htop" check
        data_path = "../input/hubmap-organ-segmentation"
        third_data_path = "../input/hubmap-2022-256x256"
        ckpt_path = "../input/ckpt-frank/resnet_img768_bs8_fold2" # for submit
        # step2: data
        n_fold = 2
        img_size = 768
        train_bs = 8
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'resnet18'
        num_classes = 1
        # step4: optimizer
        epoch = 2
        lr = 1e-3
        wd = 1e-5
        lr_drop = 8
        # step5: infer
        thr = 0.3
    
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
        
        df.loc[:,'fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df['id'], y=df['organ'])):
            df.iloc[val_idx, -1] = fold
     
        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ###############################################################
            data_transforms = {'train':train_augment, 'valid_test': valid_augment}
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms, CFG) # dataset & dtaloader
            
            model = build_model(CFG) # model
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr) # optimizer
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss

            best_val_dice = 0
            best_epoch = 0
            
            
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                # lr_scheduler.step()
                val_dice, images, probabilitys, masks, valid_id = valid_one_epoch(model, valid_loader, CFG)
                
                ###############################################################
                ##### >>>>>>> step4: save best & last model <<<<<<
                ###############################################################
                is_best = (val_dice > best_val_dice)
                best_val_dice = max(best_val_dice, val_dice)
                if is_best:
                    save_path = f"{CFG.ckpt_path}/best_fold{fold}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                ###############################################################
                ##### >>>>>>> step5: visual last epoch pred <<<<<<
                ###############################################################
                if epoch == CFG.epoch:
                    save_path = f"{CFG.ckpt_path}/last_fold{fold}.pth"
                    torch.save(model.state_dict(), save_path)
                    
                    val_img_num = images.shape[0]
                    for idx in range(10):
                        plot_visual(images[idx].cpu().numpy(), masks[idx].cpu().numpy()*255, probabilitys[idx].cpu().numpy(), valid_id[idx], "bwr")
                
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

        