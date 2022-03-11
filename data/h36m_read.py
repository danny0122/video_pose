import json
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import os.path as osp


from lib.data_utils._kp_utils import convert_kps
from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params

VIS_THRESH = 0.3


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord



def load_data(data_path,is_train=True):

    print("loading h36m data")

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
    #    'features': [],
    }

    #model = spin.get_pretrained_hmr()

    if is_train:
        subjects=[1,5,6,7,8]
    else:
        subjects=[9,11]

    annot_path = osp.join(data_path,"annotations")
    images_path = osp.join(data_path,"images")



    for subject in subjects:
        #with open( osp.join(annot_path,f"Human36M_subject{subject}_data.json"),"r") as f:
        #    annotations = json.load(f)
        with open( osp.join(annot_path,f"Human36M_subject{subject}_camera.json"),"r") as f:
            cameras = json.load(f)
        with open( osp.join(annot_path,f"Human36M_subject{subject}_joint_3d.json"),"r") as f:
            joints = json.load(f)

        with open( osp.join(annot_path,f"Human36M_subject{subject}_SMPL_NeuralAnnot.json"),"r") as f:
            smpl_params = json.load(f)


        #camera 변수들을 numpy array로 바꿔두기
        for cam_i, cam_param in cameras.items(): #Rtfc
            cameras[cam_i]["cam_param"] = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
        
        #s_01_act_02_subact_01_ca_01
        #s_01_act_02_subact_01_ca_01_000001.jpg
        print(f" subject {subject} : reading image")
        video_path_list = sorted(glob.glob( osp.join( images_path, f"s_{subject:02d}*")))

        
        for video_path in tqdm(video_path_list):
            
            video_dir = video_path.split('/')[-1]
            act = str(int(video_dir.split('_act_')[-1][0:2]))
            subact = str(int(video_dir.split('_subact_')[-1][0:2]))
            cam = str(int(video_dir.split('_ca_')[-1][0:2]))
            #폴더명 읽고 act/subact/cam 변수 읽기

            img_paths = sorted(glob.glob( osp.join( video_path, f"*.jpg" )))
            num_frames = len(img_paths)
            
            if num_frames < 1:
                continue
            
            R, t, f, c = cameras[cam]["cam_param"]
            #cam_param = cameras[cam]
            #R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
            #    cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)

            poses = np.zeros((num_frames, 72), dtype=np.float32)
            shapes = np.zeros((num_frames, 10), dtype=np.float32)
            j3ds = np.zeros((num_frames, 49, 3), dtype=np.float32)
            j2ds = np.zeros((num_frames, 49, 3), dtype=np.float32)


            for img_i in range(num_frames):
                
                smpl_param = smpl_params[act][subact][str(img_i)]
                pose = np.array(smpl_param['pose'], dtype=np.float32)
                shape = np.array(smpl_param['shape'], dtype=np.float32)

                joint_world = np.array(joints[act][subact][str(img_i)], dtype=np.float32)

                # 왜 하는거지?
                # match right, left
                match = [[1, 4], [2, 5], [3, 6]]
                for m in match:
                    l, r = m
                    joint_world[l], joint_world[r] = joint_world[r].copy(), joint_world[l].copy()

                joint_cam = world2cam(joint_world, R, t)
                joint_img = cam2pixel(joint_cam, f, c)
                #joint_world : 절대 좌표계에서 joint 좌표
                #joint_cam : 카메라 좌표계에서 joint 좌표
                #joint_img : 카메라에 사영된 joint의 2차원 좌표 

                #spin에서 사용된 joint(key point)로 바꿔주기
                j3d = convert_kps(joint_cam[None, :, :] / 1000, "h36m", "spin").reshape((-1, 3))
                j3d = j3d - j3d[39]  # 4 is the root

                joint_img[:, 2] = 1
                j2d = convert_kps(joint_img[None, :, :], "h36m", "spin").reshape((-1,3))

                poses[img_i] = pose
                shapes[img_i] = shape
                j3ds[img_i] = j3d
                j2ds[img_i] = j2d

            #영상에서 부드럽게 움직이는 bbox를 얻는 역할인듯
            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2ds, vis_thresh=VIS_THRESH, sigma=8)


            c_x = bbox_params[:, 0]
            c_y = bbox_params[:, 1]
            scale = bbox_params[:, 2]

            w = h = 150. / scale
            w = h = h * 0.9  # 1.1 for h36m_train_25fps_occ_db.pt
            bbox = np.vstack([c_x, c_y, w, h]).T

            img_paths_array = np.array(img_paths)[time_pt1:time_pt2][::2]
            bbox = bbox[::2]
            # subsample frame to 25 fps

            #dataset['vid_name'].append(np.array([f'{video_path}_{subject}'] * num_frames)[time_pt1:time_pt2][::2])
            dataset['vid_name'].append(np.array([f'{video_path}'] * num_frames)[time_pt1:time_pt2][::2])
            dataset['frame_id'].append(np.arange(0, num_frames)[time_pt1:time_pt2][::2])
            dataset['joints3D'].append(j3ds[time_pt1:time_pt2][::2])
            dataset['joints2D'].append(j2ds[time_pt1:time_pt2][::2])
            dataset['shape'].append(shapes[time_pt1:time_pt2][::2])
            dataset['pose'].append(poses[time_pt1:time_pt2][::2])

            dataset['img_name'].append(img_paths_array)
            dataset['bbox'].append(bbox)

            #features = extract_features(model, None, img_paths_array, bbox,kp_2d=j2ds[time_pt1:time_pt2][::2], debug=debug, dataset='h36m', scale=1.0)  # 1.2 for h36m_train_25fps_occ_db.pt

            #dataset['features'].append(features)

            #import pdb;pdb.set_trace()
            """
            debug_img=cv2.imread(img_paths_array[0])
            debug_img=cv2.rectangle(debug_img, (int(bbox[0][0]-bbox[0][2]/2),int(bbox[0][1]-bbox[0][3]/2)), (int(bbox[0][0]+bbox[0][2]/2),int(bbox[0][1]+bbox[0][3]/2)), (0,255,0), 3)
            cv2.imwrite("debug.png",debug_img)
            """
        
        
    
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)
    

    #import pdb;pdb.set_trace()

    return dataset
