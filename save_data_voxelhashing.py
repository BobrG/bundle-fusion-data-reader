import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# TODO: move to utils
# calibration utils
import scipy
import collections 
from scipy.spatial.transform import Rotation as rot
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image_colmap(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image_colmap(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def from_colmap(images_txt):
    colmap_images = read_images_text(images_txt)
    world_to_cam = np.zeros((len(colmap_images), 4, 4))
    world_to_cam[:, 3, 3] = 1
    index = dict()
    images = dict()
    for pos_i, (i, img) in enumerate(colmap_images.items()):
        index[i] = pos_i
        images[img.name] = pos_i
        world_to_cam[pos_i, :3, 3] = img.tvec
        wxyz = img.qvec
        xyzw = np.roll(wxyz, -1)
        rot = scipy.spatial.transform.Rotation.from_quat(xyzw).as_matrix()
        world_to_cam[pos_i, :3, :3] = rot
    return world_to_cam, index, images

def make_matrix(data):
    return np.array([np.array(d, dtype=np.float32) for d in data[1:]]).reshape((4, 4))

# depth utils
def unpack_float32(ar):
    r"""Unpacks an array of uint8 quadruplets back to the array of float32 values.
    Parameters
    ----------
    ar : np.naddary
        of shape [**, 4].
    Returns
    -------
    ar : np.naddary
        of shape [**]
    """
    shape = ar.shape[:-1]
    return ar.ravel().view(np.float32).reshape(shape)
# TODO: ends here

# main functions
def load_depthmap(file):
    r"""Loads a float32 depthmap packed into RGBA png.
    Parameters
    ----------
    file : str
    Returns
    -------
    depthmap : np.ndarray
        of shape [height, width], np.float32
    """
    depthmap = Image.open(file)
    depthmap = np.asarray(depthmap)
    depthmap = unpack_float32(depthmap)
    return depthmap

# header reader
def read_info(filename, verbose=False):
    header_data = {}
    with open(filename) as file:
        for line in file:
            print(line)
            tmp = line.strip().split('=')
            if 'Intrinsic' in tmp[0]:
                tmp_arr = np.array(tmp[-1].rstrip().split(' ')[1:], dtype=np.float32)
                intr_mat = np.eye(4, dtype=np.float32)
                intr_mat[0, 0] = tmp_arr[2]
                intr_mat[1, 1] = tmp_arr[3]
                intr_mat[0, 2] = tmp_arr[4]
                intr_mat[1, 2] = tmp_arr[5]
                header_data[tmp[0].rstrip()] = intr_mat
            elif 'Extrinsic' in tmp[0]:
                header_data[tmp[0].rstrip()] = make_matrix(tmp[-1].rstrip().split(' '))
            else:
                header_data[tmp[0].rstrip()] = tmp[-1].strip()
                
            if verbose:
                print(f'{tmp[0].rstrip()}: {header_data[tmp[0].rstrip()]}')
                    
    return header_data

# read scans
def read_frames_sk3d(folder, num_frames=100, obj='amber_vase', use_images=False, light='ambient@best', mode=3, verbose=False):
    frames = {'m_colorCompressed': [], 'm_depthCompressed': [], 'm_cameraToWorld': [],
              'm_timeStampColor': [], 'm_timeStampDepth': [], 
              'm_colorSizeBytes': [], 'm_depthSizeBytes': []}
    frames['num_frames'] = int(num_frames)
    
    folder_frames = folder + '/processed_scans/images/'
    for i in range(frames['num_frames']):
        if use_images:
            im = read_image(folder_frames + f'undist/{obj}/tis_right/rgb/{light}/{i:04d}.png')
        else:
            im = np.ones_like(im)
        
        d = load_depthmap(folder_frames + f'depthmaps/{obj}/stl@tis_right.undist/{i:04d}.png')
        if mode == 8:
#             d[np.isnan(d)] = 1.0
            d = (d*65535).astype(np.uint16)
    
        frames['m_colorCompressed'].append(im)
        frames['m_timeStampColor'].append(i)
        frames['m_depthCompressed'].append(d)
        frames['m_timeStampDepth'].append(i)
        
        h, w, c = frames['m_colorCompressed'][-1].shape
        frames['m_colorSizeBytes'].append(h*w*c) # uint8 * h * w * c 
        h, w = frames['m_depthCompressed'][-1].shape
        frames['m_depthSizeBytes'].append(2*h*w) # uint16 * h * w
    
    camposes_path = 
    world_to_cam, index, images = from_colmap(camposes_path) # (folder + '/calibration/undist_extrinsics/_calibration/tis_right.images.txt')
    frames['m_cameraToWorld'] = [np.linalg.inv(mat) for mat in world_to_cam]
    if verbose:
        print('m_cameraToWorld:', frames['m_cameraToWorld'])
   
    return frames

# write scans
def writeRGBFramesToFile_sk3d(out, rgbdframes):

    for i in tqdm(range(rgbdframes['num_frames'])):
        np.array(rgbdframes['m_cameraToWorld'][i], dtype=np.float32).tofile(out)
        np.array(rgbdframes['m_timeStampColor'][i], dtype=np.uint64).tofile(out)
        np.array(rgbdframes['m_timeStampDepth'][i], dtype=np.uint64).tofile(out)
        np.array(rgbdframes['m_colorSizeBytes'][i], dtype=np.uint64).tofile(out)
        np.array(rgbdframes['m_depthSizeBytes'][i], dtype=np.uint64).tofile(out)
        
        rgbdframes['m_colorCompressed'][i].tofile(out)
        rgbdframes['m_depthCompressed'][i].tofile(out)
        
        
def writeBinaryDumpFile(out, info, rgbdframes):
    np.array([2], dtype=np.uint32).tofile(out) # M_CALIBRATED_SENSOR_DATA_VERSION=2 in voxel hashing

    pass