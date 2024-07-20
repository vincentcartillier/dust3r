import sys
sys.path.append("/nethome/abati7/flash/Work/recon/dust3r/dependencies/FSGS/")
import torch
import numpy as np
from dust3r.model_w3dgs import AsymmetricCroCo3DStereo3DGS
from dust3r.utils.device import to_cpu, collate_with_cat
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# From FSGS package
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from arguments import PipelineParams
import torchvision.transforms as tvf


ImgNorm = tvf.Compose([tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
inf = float('inf')

# model = AsymmetricCroCo3DStereo3DGS(pos_embed='RoPE100', 
#                                     patch_embed_cls='ManyAR_PatchEmbed', 
#                                     img_size=(224, 224), 
#                                     head_type='dpt_3dgs', 
#                                     output_mode='pts3d', 
#                                     depth_mode=('exp', -inf, inf), 
#                                     conf_mode=('exp', 1, inf), 
#                                     enc_embed_dim=1024, 
#                                     enc_depth=24, 
#                                     enc_num_heads=16, 
#                                     dec_embed_dim=768, 
#                                     dec_depth=12, 
#                                     dec_num_heads=12).cuda()

img1 = torch.rand((3,224,224)).cuda()
img2 = torch.rand((3,224,224)).cuda()
imgs = []
imgs.append((dict(img=ImgNorm(img1)[None], true_shape=np.int32(
            [[224,224]]), idx=0, instance='0'),
            
            dict(img=ImgNorm(img2)[None], true_shape=np.int32(
            [[224,224]]), idx=1, instance='1')))

w2c = np.random.rand(4,4)
R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
T = w2c[:3, 3]

viewpoint_cam = Camera(colmap_id=0, R=R, T=T, 
                FoVx=np.pi/3, FoVy=np.pi/3,  image=img1, gt_alpha_mask=None,
                uid=0, data_device="cuda:0", image_name='testing',
                depth_image=None, mask=None, bounds=None) #most of these are not needed

view1, view2 = collate_with_cat(imgs)
# responses = model(view1, view2, viewpoint_cam, viewpoint_cam) #TODO: need to change 2nd viewpointCam

# Loss
# gt_image = viewpoint_cam.original_image.cuda()
# Ll1 = l1_loss_mask(responses['images'][0], gt_image)
# Ll1.backward()
print("works!")

def train_model_single_image(num_epochs, start_epoch_number=0):
    # writer = SummaryWriter(output_dir)

    model = AsymmetricCroCo3DStereo3DGS(pos_embed='RoPE100', 
                                    patch_embed_cls='ManyAR_PatchEmbed', 
                                    img_size=(224, 224), 
                                    head_type='dpt_3dgs', 
                                    output_mode='pts3d', 
                                    depth_mode=('exp', -inf, inf), 
                                    conf_mode=('exp', 1, inf), 
                                    enc_embed_dim=1024, 
                                    enc_depth=24, 
                                    enc_num_heads=16, 
                                    dec_embed_dim=768, 
                                    dec_depth=12, 
                                    dec_num_heads=12)
    lr = 0.0001
    optimizer_gaze = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr)
    # if loadPath is not None:
    #     model.load_state_dict(torch.load(loadPath, map_location="cpu"))
    # model = nn.DataParallel(model, device_ids = [0,1])
    
    model.cuda()

    # writeNum = len(train_loader) // 150
    
    # steps = start_epoch_number*len(train_loader)
    for epoch in tqdm(range(start_epoch_number, start_epoch_number+num_epochs)):

        sum_loss = 0

        # for i, data in enumerate(tqdm(train_loader)): # we can implement later
            # images_gaze, headMaps, gt_heatmap, gt_onehot_heatmap, gt_angle, gaze_cones, imgPaths = data

        responses = model(view1, view2, viewpoint_cam, viewpoint_cam) #TODO: need to change 2nd viewpointCam

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss_mask(responses['images'][0], gt_image)
        # sum_loss += Ll1.detach().cpu()

        optimizer_gaze.zero_grad(set_to_none=True)
        Ll1.backward()
        optimizer_gaze.step()
        if epoch % 20 == 0:
            print(f"Loss @ Epoch {epoch}: {Ll1}")
    # if save:
    #     if epoch % 1 == 0: #epoch-level train loss logging on tensorboard
    #         print('Taking snapshot...',
    #             torch.save(model.module.state_dict(),f"{save_dir}/_epoch_{epoch+1}_loss_{sum_loss/iter_gaze:.2f}.pkl"))
    #         writer.add_scalar("Epoch Loss-Train", sum_loss/iter_gaze, epoch+1)
    # if test:
    #     #epoch-level test loss logging on tensorboard
    #     sth = test_model(model, test_loader)
    #     writer.add_scalar("Epoch Angle Error Avg-Test", avgavg, epoch+1)

train_model_single_image(60)