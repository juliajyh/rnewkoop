from __future__ import division
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
import math as m

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = tf.cast(1. / max_depth, 'float32')
    max_disp = tf.cast(1. / min_depth, 'float32')
    scaled_disp = min_disp + (max_disp - min_disp) * disp

    depth = tf.cast(1. / scaled_disp, 'float32')
    return depth

def colorize(value, vmin=None, vmax=None, cmap=None):
    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def deg2rad(deg):
    return deg*m.pi/180

def getRotationMat(roll, pitch, yaw):

    rx = np.array([1., 0., 0., 0., np.cos(deg2rad(roll)), -np.sin(deg2rad(roll)), 0., np.sin(deg2rad(roll)), np.cos(deg2rad(roll))]).reshape((3, 3))
    ry = np.array([np.cos(deg2rad(pitch)), 0., np.sin(deg2rad(pitch)), 0., 1., 0., -np.sin(deg2rad(pitch)), 0., np.cos(deg2rad(pitch))]).reshape((3, 3))
    rz = np.array([np.cos(deg2rad(yaw)), -np.sin(deg2rad(yaw)), 0., np.sin(deg2rad(yaw)), np.cos(deg2rad(yaw)), 0., 0., 0., 1.]).reshape((3, 3))

    return np.matmul(rz, np.matmul(ry, rx))

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    print(depth.shape)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]

    depth = depth
    return depth

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = tf.reduce_max(x)
    mi = tf.reduce_min(x)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

# added by fangloveskari 2019/09/26
# adapted from monodepth2
def getTransMatrix(trans_vec):
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    batch_size= tf.shape(trans_vec)[0]
    # [B, 1, 1]
    one = tf.ones([batch_size,1,1], dtype=tf.float32)
    zero = tf.zeros([batch_size,1,1], dtype=tf.float32)

    T = tf.concat([
        one, zero, zero, trans_vec[:, :, :1],
        zero, one, zero, trans_vec[:, :, 1:2],
        zero, zero, one, trans_vec[:, :, 2:3],
        zero, zero, zero, one

    ], axis=2)

    T = tf.reshape(T,[batch_size, 4, 4])


    # T = tf.zeros([trans_vec.get_shape().as_list()[0],4,4],dtype=tf.float32)
    # for i in range(4):
    #     T[:,i,i] = 1
    # trans_vec = tf.reshape(trans_vec, [-1,3,1])
    # T[:,:3,3] = trans_vec
    return T

def rotFromAxisAngle(vec):
    """
    Convert axis angle into rotation matrix
    not euler angle but Axis
    :param vec: [B, 1, 3]
    :return:
    """
    angle = tf.norm(vec,ord=2,axis=2,keepdims=True)
    axis = vec / (angle + 1e-7)

    ca = tf.cos(angle)
    sa = tf.sin(angle)

    C = 1 - ca

    x = axis[:,:,:1]
    y = axis[:,:,1:2]
    z = axis[:,:,2:3]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # [B, 1, 1]
    one = tf.ones_like(zxC, dtype=tf.float32)
    zero = tf.zeros_like(zxC, dtype=tf.float32)

    rot_matrix = tf.concat([
        x * xC + ca, xyC - zs, zxC + ys, zero,
        xyC + zs, y * yC + ca, yzC - xs, zero,
        zxC - ys, yzC + xs, z * zC + ca, zero,
        zero, zero, zero, one
    ],axis=2)

    rot_matrix = tf.reshape(rot_matrix, [-1,4,4])

    return rot_matrix

def pose_axis_angle_vec2mat(vec, invert=False):
    """
    Convert axis angle and translation into 4x4 matrix
    :param vec: [B,1,6] with former 3 vec is axis angle
    :return:
    """
    batch_size = tf.shape(vec)[0]
    axisvec = tf.slice(vec, [0, 0], [-1, 3])
    axisvec = tf.reshape(axisvec, [batch_size, 1, 3])

    translation = tf.slice(vec, [0, 3], [-1, 3])
    translation = tf.reshape(translation, [batch_size, 1, 3])


    R = rotFromAxisAngle(axisvec)

    if invert:
        R = tf.transpose(R, [0,2,1])
        translation *= -1
    t = getTransMatrix(translation)

    if invert:
        M = tf.matmul(R,t)
    else:
        M = tf.matmul(t,R)
    return M

def euler2mat(z, y, x):
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size = tf.shape(vec)[0]
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.
  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  _, height, width, _ = depth.get_shape().as_list()
  batch = tf.shape(depth)[0]
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.linalg.inv(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.
  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  _, _, height, width = cam_coords.get_shape().as_list()
  batch = tf.shape(cam_coords)[0]
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.
  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))

  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def partial_disp(raw_disp, f, params, height=200, width=200):
    u0 = height / 2.
    v0 = width / 2.
    batch = tf.shape(raw_disp)[0]
    grids = meshgrid(batch, height, width)
    grid_x = grids[:, 0]
    grid_y = grids[:, 1]
    disp_maps = []
    # params = list(np.linspace(0, 2., num))
    # print(params)
    f_x = tf.expand_dims(f * width / 416., 2)
    f_y = tf.expand_dims(f * height / 128., 2)
    for i, start_idx in enumerate(params):
        X_Cam = tf.divide(grid_x - u0, tf.tile(f_x, [1, height, width]))
        Y_Cam = tf.divide(grid_y - v0, tf.tile(f_y, [1, height, width]))

        AuxVal = tf.multiply(X_Cam, X_Cam) + tf.multiply(Y_Cam, Y_Cam)

        xi = 0
        alpha_cam = xi + tf.math.sqrt(1 + AuxVal)
        alpha_div = AuxVal + 1

        alpha_cam_div = tf.divide(alpha_cam, alpha_div)

        X_Sph = tf.multiply(X_Cam, alpha_cam_div)
        Y_Sph = tf.multiply(Y_Cam, alpha_cam_div)
        Z_Sph = alpha_cam_div - xi

        r = np.matmul(getRotationMat(0, 0, 0), np.matmul(getRotationMat(0, -90, 45), getRotationMat(0, 90, 90)))

        elems1 = r[:, 0]
        elems2 = r[:, 1]
        elems3 = r[:, 2]

        x1 = elems1[0] * X_Sph + elems2[0] * Y_Sph + elems3[0] * Z_Sph
        y1 = elems1[1] * X_Sph + elems2[1] * Y_Sph + elems3[1] * Z_Sph
        z1 = elems1[2] * X_Sph + elems2[2] * Y_Sph + elems3[2] * Z_Sph

        X_Sph = x1
        Y_Sph = y1
        Z_Sph = z1

        # 4. cart 2 sph
        ntheta = tf.math.atan2(Y_Sph, X_Sph)
        nphi = tf.math.atan2(Z_Sph, tf.math.sqrt(tf.multiply(X_Sph, X_Sph) + tf.multiply(Y_Sph, Y_Sph)))

        pi = m.pi

        min_phi = -pi / 10.
        max_phi = pi / 10.

        min_theta = -pi / 3. + start_idx
        max_theta = pi + start_idx

        ImPano_W = raw_disp.get_shape()[2]
        ImPano_H = raw_disp.get_shape()[1]

        min_x = 0
        max_x = ImPano_W - 1.0
        min_y = 0
        max_y = ImPano_H - 1.0

        ## for x
        a = (max_theta - min_theta) / (max_x - min_x)
        b = max_theta - a * max_x  # from y=ax+b %% -a;
        nx = (1. / a) * (ntheta - b)

        ## for y
        a = (min_phi - max_phi) / (max_y - min_y)
        b = max_phi - a * min_y  # from y=ax+b %% -a;
        ny = (1. / a) * (nphi - b)
        # 6. Final step interpolation and mapping
        im, _ = bilinear_sampler(batch, raw_disp, tf.stack([nx, ny], -1))
        disp_maps.append(im)

    return tf.stack(disp_maps, 1)


def projective_inverse_warp(img, depth, pose, intrinsics,invert=False, euler=False):
  """Inverse warp a source image to the target image plane based on projection.
  Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of rx, ry, rz, tx, ty, tz
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
  """
  batch = tf.shape(img)[0]
  height = tf.shape(img)[1]
  width = tf.shape(img)[2]
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  output_img, _ = bilinear_sampler(batch, img, src_pixel_coords)
  return output_img

def bilinear_sampler(batch, imgs, coords):
  """Construct a new image by bilinear sampling from the input image.
  Points falling outside the source image boundary have value 0.
  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    imgs = tf.reshape(imgs, [batch, tf.shape(imgs)[-3], tf.shape(imgs)[-2], -1])
    inp_size = [tf.shape(imgs)[i] for i in range(4)]
    coord_size = coords.get_shape()
    out_size = [tf.shape(coords)[0], tf.shape(coords)[1], tf.shape(coords)[2], tf.shape(imgs)[3]]
    # out_size = coords.get_shape().as_list()
    # out_size[3] = tf.shape(imgs)[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')


    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')
    eps = tf.constant([0.5], tf.float32)

    coords_x = tf.clip_by_value(coords_x, eps, x_max - eps)
    coords_y = tf.clip_by_value(coords_y, eps, y_max - eps)

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x # 1
    wt_x1 = coords_x - x0_safe # 0
    wt_y0 = y1_safe - coords_y # 1
    wt_y1 = coords_y - y0_safe # 0

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(tf.shape(coords)[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [tf.shape(coords)[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output, w00 + w01 + w10 + w11

def load_resnet18_from_file(res18_file):
    res18_weights = np.load(res18_file,allow_pickle=True).item()
    all_vars = tf.global_variables()
    assign_ops = []
    for v in all_vars:
        if v.op.name == 'global_step' or 'encoder' not in v.op.name or 'Adam' in v.op.name:
            continue
        print('\t' + v.op.name)
        if 'pose_encoder/conv1/conv1/' in v.op.name:
            new_op_name = v.op.name.replace('pose_','')
            pose_conv1_weight = np.concatenate((res18_weights[new_op_name],res18_weights[new_op_name]), axis=2) / 2
            assign_op = v.assign(pose_conv1_weight)
        elif 'pose_encoder' in v.op.name:
            new_op_name = v.op.name.replace('pose_','')
            assign_op = v.assign(res18_weights[new_op_name])
        else:
            assign_op = v.assign(res18_weights[v.op.name])


        #assign_op = v.assign(res18_weights[v.op.name])
        assign_ops.append(assign_op)
    return assign_ops