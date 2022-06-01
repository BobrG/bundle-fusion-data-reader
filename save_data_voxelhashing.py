import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

from pathfinder import Pathfinder, sensor_to_cam_mode # TODO: rewrite to utils.pathfinder

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

def extract_pinhole(filename):
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                continue
            # line looks like: 0 PINHOLE 1872 1024 1394.6050886373168 1401.287173228876 935.753089080528 512.1488159189286
            tmp_arr = np.array(line.rstrip().split(' '), dtype=np.float32)
            intr_mat = np.eye(4, dtype=np.float32)
            h, w = tmp_arr[2], tmp_arr[3]
            intr_mat[0, 0] = tmp_arr[4]
            intr_mat[1, 1] = tmp_arr[5]
            intr_mat[0, 2] = tmp_arr[6]
            intr_mat[1, 2] = tmp_arr[7]

    return intr_mat, h, w

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
# filename: {sensorname}_voxelhashing_info.txt
def read_info(folder, sensor='kinect', verbose=False):
    header_data = {}
    sensordata = Pathfinder(data_root=f'{folder}/dataset/', raw_scans_root=f'{folder}/raw_scans/')[sensor]
    
    # TODO: make function for intr, extr matrices extraction
#     intr_mat, h, w = extract_pinhole(sensordata['calibrated_intrinsics'])
    intr_mat = np.eye(3)
    header_data['m_calibrationColorIntrinsic'] = intr_mat # TODO: find default parameters # make_pinhole_intrmat(sensordata['pinhole_intrinsics'])
    header_data['m_calibrationColorExtrinsic'] = np.eye(4) # sensordata['calibrated_extrinsics']
    header_data['m_colorWidth'] = w
    header_data['m_colorHeight'] = h
    
#     intr_mat, h, w = extract_pinhole(sensordata['calibrated_intrinsics'])
    header_data['m_calibrationDepthIntrinsic'] = intr_mat # TODO: find default parameters # make_pinhole_intrmat(sensordata['pinhole_intrinsics'])
    header_data['m_calibrationDepthExtrinsic'] = np.eye(4) # sensordata['calibrated_extrinsics']
    header_data['m_depthWidth'] = w
    header_data['m_depthHeight'] = h
    
    # TODO: for BundleFusion define DepthShift, SensorName and versionNumber
    
    if verbose:
        print('header dict:', header_data)
                    
    return header_data

# read scans
# folder: contains dataset/ and raw_scans/ subfolders, is used for creating pathfinder
def read_frames_sk3d(folder, num_frames=100, 
                     sensor='kinect', obj='amber_vase', 
                     use_images=False, light='ambient@best', 
                     mode=3, verbose=False):
    
    if not ('dataset' in os.listdir(folder) and 'raw_scans' in os.listdir(folder)):
        print('Wrong path!')
        return None
    
    sensordata = Pathfinder(data_root=f'{folder}/dataset/', raw_scans_root=f'{folder}/raw_scans/')[sensor]
    frames = {'m_colorCompressed': [], 'm_depthCompressed': [], 'm_cameraToWorld': [],
              'm_timeStampColor': [], 'm_timeStampDepth': [], 
              'm_colorSizeBytes': [], 'm_depthSizeBytes': []}
    frames['num_frames'] = int(num_frames)
    
    for i in range(frames['num_frames']):
        if use_images:
            im = read_image(sensordata['rgb'].undistorted[light, i]) # TODO: check whether undistorted is correct option
        else:
            im = np.ones_like(im)
        
        depth = load_depthmap(sensordata['depth'].raw[i])
        if mode == 8:
#             d[np.isnan(d)] = 1.0
            d = (d*65535).astype(np.uint16)
    
        frames['m_colorCompressed'].append(im)
        frames['m_timeStampColor'].append(i)
        frames['m_depthCompressed'].append(depth)
        frames['m_timeStampDepth'].append(i)
        
        h, w, c = frames['m_colorCompressed'][-1].shape
        frames['m_colorSizeBytes'].append(h*w*c) # uint8 * h * w * c 
        h, w = frames['m_depthCompressed'][-1].shape
        frames['m_depthSizeBytes'].append(2*h*w) # uint16 * h * w
    
    camposes_path = sensordata['depth']['calibrated_extrinsics']
    world_to_cam, index, images = from_colmap(camposes_path)
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

    np.array(rgbdframes['num_frames'], np.uint32).tofile(out)
    np.array(info['m_depthWidth'], dtype=np.uint32).tofile(out)
    np.array(info['m_depthHeight'], dtype=np.uint32).tofile(out)
    
    np.array(rgbdframes['num_frames'], np.uint32).tofile(out)
    np.array(info['m_colorWidth'], dtype=np.uint32).tofile(out)
    np.array(info['m_colorHeight'], dtype=np.uint32).tofile(out)
    
    info['m_calibrationDepthIntrinsic'].tofile(out)
    np.linalg.inv(info['m_calibrationDepthIntrinsic']).tofile(out)
    info['m_calibrationDepthExtrinsic'].tofile(out)
    np.linalg.inv(info['m_calibrationDepthExtrinsic']).tofile(out)
    
    info['m_calibrationColorIntrinsic'].tofile(out)
    np.linalg.inv(info['m_calibrationColorIntrinsic']).tofile(out)
    info['m_calibrationColorExtrinsic'].tofile(out)
    np.linalg.inv(info['m_calibrationColorExtrinsic']).tofile(out)

    for i in range(rgbdframes['num_frames']):
        rgbdframes['m_depthCompressed'][i].tofile(out)
        
    for i in range(rgbdframes['num_frames']):
        ones = np.ones((int(info['m_colorWidth'])*int(info['m_colorHeight']), 1), dtype=np.uint8)
        rgbw_image = np.concatenate((rgbdframes['m_colorCompressed'][i].reshape((int(info['m_colorWidth'])*int(info['m_colorHeight']), 3)),
                             ones), axis=1)
        np.array(rgbw_image).tofile(out)
        
    np.array(rgbdframes['num_frames'], dtype=np.uint64).tofile(out)

    for i in range(rgbdframes['num_frames']):
         np.array(rgbdframes['m_timeStampColor'][i], dtype=np.uint64).tofile(out)
    
    np.array(rgbdframes['num_frames'], dtype=np.uint64).tofile(out)
    for i in range(rgbdframes['num_frames']):
         np.array(rgbdframes['m_timeStampDepth'][i], dtype=np.uint64).tofile(out)
            
    np.array(rgbdframes['num_frames'], dtype=np.uint64).tofile(out)
    for i in range(rgbdframes['num_frames']):
         np.array(rgbdframes['m_cameraToWorld'][i], dtype=np.float32).tofile(out)

# aggregate all functions above            
def saveToFile(data_dir, filename_out, read_frames, write_frames, dir_out='./', mode=8):
    num_poses = 100 # TODO: where can I get this number from file?
    
    if mode == 8: # SensorDataReader 
        raise NotImplemented
        
    elif mode == 7 or mode == 3: # BinaryDumpReader in BundleFusion or in VoxelHashing
        out = open(f'{dir_out}/{filename_out}.sensor', 'wb')
        data_header = read_info(data_dir)
        rgbdframes = read_frames(data_dir, num_frames=num_poses, mode=mode)#read rgbframes from data dir

        writeBinaryDumpFile(out, data_header, rgbdframes)