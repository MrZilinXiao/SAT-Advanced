import os
import torch
import clip
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device('cuda:1')
# device = "cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)
# # model, preprocess = clip.load("RN50x4", device=device)
model, preprocess = clip.load("RN50x16", device=device)

model.eval()  # newly added

# clipdim=512
clipdim = 768

# # image = preprocess(Image.open("CLIP.png",'r')).unsqueeze(0).to(device)
# # image = torch.from_numpy(np.array(Image.open("CLIP.png",'r').convert('RGB'),dtype=np.float32))
# # image = torch.tensor(Image.open("CLIP.png",'r').convert('RGB'))
# image = loader(Image.open("CLIP.png",'r').convert('RGB'))
# print(image,image.shape) ## torch.Size([3, 762, 2162]) D, H, W
# image = preprocess(image).unsqueeze(0).to(device)
# print(image)
# print(image.shape) ## torch.Size([1, 3, 224, 224]) torch.Size([1, 3, 288, 288]), torch.Size([1, 3, 384, 384])
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# # print(text) ## torch.Size([1, 640]) torch.Size([3, 640])

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     print(image_features.shape,text_features.shape)
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


# output_path = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_CLIP_VITB16'
output_path = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_CLIP_RN5016'
# output_scale_path = '/localdisk2/zyang39/DATASET/scannet/tasks/scannet_frames_25k_CLIP_VITB16_15x'

# context_path = '/localdisk2/zyang39/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat_aggregate'
context_path = '/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat_aggregate_frameid'
filelist = os.listdir(context_path)
loader = transforms.Compose([transforms.ToTensor()])  # necessary? 255->1

for scan_id in tqdm(filelist):
    scan_id = scan_id[:-4]
    save_path = os.path.join(output_path, '%s.npy' % scan_id)
    context_2d = np.load('%s/%s.npy' % (context_path, scan_id), allow_pickle=True,
                         encoding='latin1')  ## TODO: update relative path
    imagefolder_path = os.path.join('/localdisk2/DATASET/scannet/tasks/scannet_frames_25k', scan_id)
    context_bbox = context_2d.item()['obj_coord']
    frame_ids = context_2d.item()['frame_id']
    obj_feat = np.zeros((context_2d.item()['obj_feat'].shape[0], context_2d.item()['obj_feat'].shape[1], clipdim))
    scaled_obj_feat = np.zeros(
        (context_2d.item()['obj_feat'].shape[0], context_2d.item()['obj_feat'].shape[1], clipdim))
    # for sampled_id in range(0,int(1e6)): #range(0,int(1e6),100):
    # print(frame_ids)

    for sampled_id in range(obj_feat.shape[1]):
        for box_ii in range(obj_feat.shape[0]):
            frame_id = frame_ids[box_ii, sampled_id]
            image_path = os.path.join(imagefolder_path, 'color/%06d.jpg' % (
                frame_id))  ## 570, 93-2,173-1,251,705-1,412-1 has no 0th frame // so just do nothing all zero now?
            # print(scan_id,len(os.listdir(imagefolder_path+'/color')),context_2d.item()['obj_feat'].shape)
            if not os.path.isfile(image_path) and frame_id == 0:
                print(scan_id, frame_id)
                break
            image = loader(Image.open(image_path, 'r').convert('RGB'))  # torch.Tensor
            bbox = context_bbox[box_ii, sampled_id, :]
            if bbox.sum() == 0: continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if x1 == x2 or y1 == y2: continue
            region_ii = image[:, y1:y2, x1:x2]  # torch.Tensor
            region_ii = preprocess(region_ii).unsqueeze(0).to(device)
            image_features = model.encode_image(region_ii)
            obj_feat[box_ii, sampled_id, :] = image_features.detach().cpu().numpy()

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            ## get 1.5 scale up
            x1, y1, x2, y2 = max(0, (2 * x1 - x2)), max(0, (2 * y1 - y2)), min(image.shape[2], (2 * x2 - x1)), min(
                image.shape[1], (2 * y2 - y1))
            if x1 == x2 or y1 == y2: continue
            region_ii = image[:, y1:y2, x1:x2]
            region_ii = preprocess(region_ii).unsqueeze(0).to(device)
            image_features = model.encode_image(region_ii)
            scaled_obj_feat[box_ii, sampled_id, :] = image_features.detach().cpu().numpy()

        # for box_ii in range(bbox.shape[0]):
        #     if bbox[box_ii,:].sum()==0: continue
        #     x1,y1,x2,y2 = int(bbox[box_ii,0]),int(bbox[box_ii,1]),int(bbox[box_ii,2]),int(bbox[box_ii,3])
        #     region_ii = image[:,y1:y2,x1:x2]
        #     region_ii = preprocess(region_ii).unsqueeze(0).to(device)
        #     image_features = model.encode_image(region_ii)
        #     obj_feat[box_ii,sampled_id,:] = image_features.detach().cpu().numpy()
        # # print(obj_feat)
        # # print(context_2d.item()['obj_feat'][:,sampled_id,:])
        # for box_ii in range(bbox.shape[0]):
        #     if bbox[box_ii,:].sum()==0: continue
        #     x1,y1,x2,y2 = int(bbox[box_ii,0]),int(bbox[box_ii,1]),int(bbox[box_ii,2]),int(bbox[box_ii,3])
        #     ## get 1.5 scale up
        #     x1,y1,x2,y2 = max(0,(2*x1-x2)),max(0,(2*y1-y2)),min(image.shape[2],(2*x2-x1)),min(image.shape[1],(2*y2-y1))
        #     region_ii = image[:,y1:y2,x1:x2]
        #     region_ii = preprocess(region_ii).unsqueeze(0).to(device)
        #     image_features = model.encode_image(region_ii)
        #     scaled_obj_feat[box_ii,sampled_id,:] = image_features.detach().cpu().numpy()

    np.save(
        save_path, {'clip_region_feat': obj_feat, 'clip_scaled_region_feat': scaled_obj_feat, \
                    # 读下面这些的script师兄没给，需要自己写一个从解析后的sens文件夹读这些东西的脚本
                    # 可以先尝试复现一下原版SAT的效果
                    'obj_feat': context_2d.item()['obj_feat'], 'obj_coord': context_2d.item()['obj_coord'], \
                    'obj_size': context_2d.item()['obj_size'], 'camera_pose': context_2d.item()['camera_pose'], \
                    'instance_id': context_2d.item()['instance_id'], 'obj_depth': context_2d.item()['obj_depth'], \
                    'frame_id': context_2d.item()['frame_id']}
    )
