import os
import torch
import random
import numpy as np
import os.path as osp
import copy
import cv2

from torch.utils.data import Dataset


import data.h36m_read
import lib.data_utils._img_utils


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB': img = img[:,:,::-1]
    img = img.astype(np.float32)
    return img


class H36MDataset(Dataset):

    def __init__(self,is_train,seqlen=7,overlap=0):
        self.db= data.h36m_read.load_data("data/Human36M",is_train)


        self.seqlen = seqlen
        self.mid_frame = int(seqlen/2)  

        """
        self.stride = int(seqlen * (1-overlap) + 0.5)

        self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride, is_train)
        """
        
        vid_name_db=self.db["vid_name"]

        self.vid_indices=[]

        for img_idx,vid_name in enumerate(vid_name_db):
            if img_idx>len(self.db["vid_name"])-seqlen: # 0~9 / 3 / 789
                continue # 끝을 넘어감

            available_vid=1
            for frame_idx in range(seqlen):  
                # 연속된 프레임 중간에 다른 비디오로 바뀌는 경우 데이터에 포함하지 않음
                if vid_name_db[img_idx+frame_idx]!=vid_name_db[img_idx]:
                    available_vid=0
                    print(f"unavilable idx : {img_idx}")
                    break
            
            if available_vid:
                self.vid_indices.append(img_idx)






        import pdb;pdb.set_trace()

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):


        img_idx=self.vid_indices[index]

        frames=[]
        bbox_list=[]
        
        for frame_idx in range(self.seqlen):
            vid_dir=self.db["vid_name"][img_idx+frame_idx]
            img_idx_name=self.db["frame_id"][img_idx+frame_idx] + 1 # 이미지 이름은 1부터 시작
            img_name=f"{vid_dir.split('/')[-1]}_{img_idx_name:06d}.jpg"
            img_path = osp.join(vid_dir,img_name)
            print(img_path)
            img=load_img(img_path)

            frames.append(img)
            bbox_list.append(self.db["bbox"][img_idx+frame_idx])
        
        # debug gif image output
        import imageio
        with imageio.get_writer("debug.gif", mode="I") as writer:
            for idx, frame in enumerate(frames):
                bbox=bbox_list[idx]
                rgb_frame=frame
                #rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                rgb_frame=cv2.rectangle(rgb_frame, (int(bbox[0]-bbox[2]/2),int(bbox[1]-bbox[3]/2)),
                 (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2)), (0,255,0), 3)
                writer.append_data(rgb_frame)

        import pdb;pdb.set_trace()
        
        #return 





        




