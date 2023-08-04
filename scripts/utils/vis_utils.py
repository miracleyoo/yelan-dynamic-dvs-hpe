import io
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .skeleton_helpers import Skeleton

# __all__ = ['get_skeleton_lines', 'get_3d_ax', 'plot_3d', 'plot_skeleton_3d', 'plot_skeleton_2d', 'plot_2d_overlay']


def get_img_from_fig(fig, dpi=180, pad_inches=0):
    """ A function which returns an image as numpy array from plt figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_skeleton_lines(x, y, z):
    """
    From DHP19 toolbox
    """
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR = x[0], x[1], x[2], x[3]
    x_elbowL, x_hipR, x_hipL = (
        x[4],
        x[5],
        x[6],
    )
    x_handR, x_handL, x_kneeR = (
        x[7],
        x[8],
        x[9],
    )
    x_kneeL, x_footR, x_footL = x[10], x[11], x[12]

    y_head, y_shoulderR, y_shoulderL, y_elbowR = y[0], y[1], y[2], y[3]
    y_elbowL, y_hipR, y_hipL = (
        y[4],
        y[5],
        y[6],
    )
    y_handR, y_handL, y_kneeR = (
        y[7],
        y[8],
        y[9],
    )
    y_kneeL, y_footR, y_footL = y[10], y[11], y[12]

    z_head, z_shoulderR, z_shoulderL, z_elbowR = z[0], z[1], z[2], z[3]
    z_elbowL, z_hipR, z_hipL = (
        z[4],
        z[5],
        z[6],
    )
    z_handR, z_handL, z_kneeR = (
        z[7],
        z[8],
        z[9],
    )
    z_kneeL, z_footR, z_footL = z[10], z[11], z[12]

    # definition of the lines of the skeleton graph
    skeleton = np.zeros((14, 3, 2))
    skeleton[0, :, :] = [
        [x_head, x_shoulderR],
        [y_head, y_shoulderR],
        [z_head, z_shoulderR],
    ]
    skeleton[1, :, :] = [
        [x_head, x_shoulderL],
        [y_head, y_shoulderL],
        [z_head, z_shoulderL],
    ]
    skeleton[2, :, :] = [
        [x_elbowR, x_shoulderR],
        [y_elbowR, y_shoulderR],
        [z_elbowR, z_shoulderR],
    ]
    skeleton[3, :, :] = [
        [x_elbowL, x_shoulderL],
        [y_elbowL, y_shoulderL],
        [z_elbowL, z_shoulderL],
    ]
    skeleton[4, :, :] = [
        [x_elbowR, x_handR],
        [y_elbowR, y_handR],
        [z_elbowR, z_handR],
    ]
    skeleton[5, :, :] = [
        [x_elbowL, x_handL],
        [y_elbowL, y_handL],
        [z_elbowL, z_handL],
    ]
    skeleton[6, :, :] = [
        [x_hipR, x_shoulderR],
        [y_hipR, y_shoulderR],
        [z_hipR, z_shoulderR],
    ]
    skeleton[7, :, :] = [
        [x_hipL, x_shoulderL],
        [y_hipL, y_shoulderL],
        [z_hipL, z_shoulderL],
    ]
    skeleton[8, :, :] = [[x_hipR, x_kneeR], [y_hipR, y_kneeR], [z_hipR, z_kneeR]]
    skeleton[9, :, :] = [[x_hipL, x_kneeL], [y_hipL, y_kneeL], [z_hipL, z_kneeL]]
    skeleton[10, :, :] = [
        [x_footR, x_kneeR],
        [y_footR, y_kneeR],
        [z_footR, z_kneeR],
    ]
    skeleton[11, :, :] = [
        [x_footL, x_kneeL],
        [y_footL, y_kneeL],
        [z_footL, z_kneeL],
    ]
    skeleton[12, :, :] = [
        [x_shoulderR, x_shoulderL],
        [y_shoulderR, y_shoulderL],
        [z_shoulderR, z_shoulderL],
    ]
    skeleton[13, :, :] = [[x_hipR, x_hipL], [y_hipR, y_hipL], [z_hipR, z_hipL]]
    return skeleton


def get_2d_skeleton_lines(x, y):
    """
    From DHP19 toolbox
    """
    # rename joints to identify name and axis
    x_head, x_shoulderR, x_shoulderL, x_elbowR = x[0], x[1], x[2], x[3]
    x_elbowL, x_hipR, x_hipL = (
        x[4],
        x[5],
        x[6],
    )
    x_handR, x_handL, x_kneeR = (
        x[7],
        x[8],
        x[9],
    )
    x_kneeL, x_footR, x_footL = x[10], x[11], x[12]

    y_head, y_shoulderR, y_shoulderL, y_elbowR = y[0], y[1], y[2], y[3]
    y_elbowL, y_hipR, y_hipL = (
        y[4],
        y[5],
        y[6],
    )
    y_handR, y_handL, y_kneeR = (
        y[7],
        y[8],
        y[9],
    )
    y_kneeL, y_footR, y_footL = y[10], y[11], y[12]

    # definition of the lines of the skeleton graph
    skeleton = np.zeros((14, 2, 2))
    skeleton[0, :, :] = [
        [x_head, x_shoulderR],
        [y_head, y_shoulderR],
    ]
    skeleton[1, :, :] = [
        [x_head, x_shoulderL],
        [y_head, y_shoulderL],
    ]
    skeleton[2, :, :] = [
        [x_elbowR, x_shoulderR],
        [y_elbowR, y_shoulderR],
    ]
    skeleton[3, :, :] = [
        [x_elbowL, x_shoulderL],
        [y_elbowL, y_shoulderL],
    ]
    skeleton[4, :, :] = [
        [x_elbowR, x_handR],
        [y_elbowR, y_handR],
    ]
    skeleton[5, :, :] = [
        [x_elbowL, x_handL],
        [y_elbowL, y_handL],
    ]
    skeleton[6, :, :] = [
        [x_hipR, x_shoulderR],
        [y_hipR, y_shoulderR],
    ]
    skeleton[7, :, :] = [
        [x_hipL, x_shoulderL],
        [y_hipL, y_shoulderL],
    ]
    skeleton[8, :, :] = [
        [x_hipR, x_kneeR],
        [y_hipR, y_kneeR]
    ]
    skeleton[9, :, :] = [
        [x_hipL, x_kneeL],
        [y_hipL, y_kneeL]
    ]
    skeleton[10, :, :] = [
        [x_footR, x_kneeR],
        [y_footR, y_kneeR],
    ]
    skeleton[11, :, :] = [
        [x_footL, x_kneeL],
        [y_footL, y_kneeL],
    ]
    skeleton[12, :, :] = [
        [x_shoulderR, x_shoulderL],
        [y_shoulderR, y_shoulderL],
    ]
    skeleton[13, :, :] = [
        [x_hipR, x_hipL],
        [y_hipR, y_hipL]
    ]
    return skeleton


def plot_2d_dots(joints, ax=None, plot_lines=False, c="red", label=None):
    x = joints[:, 0]
    y = joints[:, 1]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
    lines_skeleton = get_2d_skeleton_lines(x, y)
    ax.plot(joints[:, 0], joints[:, 1], '.', c=c, label=label)
    if plot_lines:
        for line in range(len(lines_skeleton)):
            ax.plot(
                lines_skeleton[line, 0, :],
                lines_skeleton[line, 1, :],
                c
            )


def plot_skeleton_2d(dvs_frame, gt_joints, pred_joints=None, plot_lines=False, ret_fig=False, gt_label='GT', pred_label='Prediction'):
    """
        To plot image and 2D ground truth and prediction

        Args:
          dvs_frame: frame as vector (1xWxH)
          sample_gt: gt joints as vector (N_jointsx2)
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(dvs_frame)
    ax.axis('off')
    plot_2d_dots(gt_joints, ax, plot_lines=plot_lines, c='red', label=gt_label)
    if pred_joints is not None:
        plot_2d_dots(pred_joints, ax, plot_lines=plot_lines, c='blue', label=pred_label)
        ax.legend(fontsize=15)

    # plt.tight_layout(pad=0)
    if ret_fig:
        return fig


def plot_2d_overlay(gt_pose, intrinsic_matrix, extrinsic_matrix, image, frame_size, pred_pose=None, plot_lines=False, ret_fig=False, show=True, gt_label='GT', pred_label='Prediction'):
    h, w = frame_size

    def process_pose_to_joints(pose):
        sk = Skeleton(pose)
        joints_2d = torch.tensor(sk.get_2d_points(
            w,
            h,
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
        ))
        joints = torch.stack([joints_2d[:, 0], joints_2d[:, 1]], 1)
        return joints

    gt_joints = process_pose_to_joints(gt_pose)
    if pred_pose is not None:
        pred_joints = process_pose_to_joints(pred_pose)
        fig = plot_skeleton_2d(image, gt_joints=gt_joints, pred_joints=pred_joints, plot_lines=plot_lines, ret_fig=ret_fig, gt_label=gt_label, pred_label=pred_label)
    else:
        fig = plot_skeleton_2d(image, gt_joints, plot_lines=plot_lines, ret_fig=ret_fig, gt_label=gt_label, pred_label=pred_label)
    if not show:
        plt.close()
    if ret_fig:
        return fig


def get_3d_ax(ret_fig=False):
    fig = plt.figure(figsize=(8, 8))
    # ax = Axes3D(fig)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    ax.view_init(30, 240)
    if not ret_fig:
        return ax
    else:
        return ax, fig


def plot_3d(points, ax, fig, c="red", limits=None, plot_lines=True, angle=270, cam_height=10, title=None):
    """
    Plot the provided skeletons in 3D coordinate space
    Args:
        ax: axes for plot
        y_true_pred: joints to plot in 3D coordinate space
        c: color (Default value = 'red')
        limits: list of 3 ranges (x, y, and z limits)
        plot_lines:  (Default value = True)
        title: the title of this figure.

    Note:
        Plot the provided skeletons. Visualization purpose only

    From DHP19 toolbox
    """

    if limits is None:
        limits = [[-3, 3], [-3, 3], [0, 15]]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, zdir="z", s=20, c=c, marker="o", depthshade=True)

    lines_skeleton = get_skeleton_lines(x, y, z)

    if plot_lines:
        for line in range(len(lines_skeleton)):
            ax.plot(
                lines_skeleton[line, 0, :],
                lines_skeleton[line, 1, :],
                lines_skeleton[line, 2, :],
                c,
                label="gt",
            )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    x_limits = limits[0]
    y_limits = limits[1]
    z_limits = limits[2]
    x_range = np.abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = np.abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = np.abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * np.max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    if title is not None:
        ax.set(title=title)

    ax.view_init(elev=cam_height, azim=angle)
    # plt.tight_layout(pad=0)


def plot_skeleton_3d(points, angle=270, cam_height=10, ret_fig=False, limits=None, title=None, no_perspective=False):
    """
        Args:
           points: ndarray, (13, 3), the 13 joints' xyz coordinates. 
           angle: float, the view angle of the camera.
           cam_height: float, the camera's view point's height.
           ret_fig: bool, Whether return the processed input image.
           limits: tuple/list, the xyz axis limit of the 3d coordinates.
           title: str, the title of output figure.
        """
    ax, fig = get_3d_ax(ret_fig=True)
    plot_3d(points, ax, fig, c='red', angle=angle, cam_height=cam_height, limits=limits, title=title)
    if no_perspective:
        ax.set_proj_type('ortho')
    if ret_fig:
        return fig


def plot_2d_from_3d(dvs_frame, gt_skeleton, p_mat, pred_skeleton=None):
    """
        To plot image and 2D ground truth and prediction

        Args:
          dvs_frame: frame as vector (1xWxH)
          sample_gt: gt joints as vector (N_jointsx2)

        """

    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(dvs_frame)
    H, W = dvs_frame.shape

    gt_joints = gt_skeleton.get_2d_points(p_mat, 346, 260)
    ax.plot(gt_joints[:, 0], gt_joints[:, 1], '.', c='red', label='gt')
    if pred_skeleton is not None:
        pred_joints = pred_skeleton.get_2d_points(p_mat, 346, 260)
        ax.plot(pred_joints[:, 0], pred_joints[:, 1], '.', c='blue', label='pred')

    plt.legend()


# def plot_skeleton_2d(dvs_frame, gt_joints, pred_joints=None):
#     """
#         To plot image and 2D ground truth and prediction

#         Args:
#           dvs_frame: frame as vector (1xWxH)
#           sample_gt: gt joints as vector (N_jointsx2)

#         """

#     fig = plt.figure()
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.imshow(dvs_frame)
#     ax.axis('off')
#     # H, W = dvs_frame.shape
#     ax.plot(gt_joints[:, 0], gt_joints[:, 1], '.', c='red')
#     if pred_joints is not None:
#         ax.plot(pred_joints[:, 0], pred_joints[:, 1], '.', c='blue')
#         plt.legend(['GT', 'Prediction'])


# def plot_2d_overlay(gt_pose, intrinsic_matrix, extrinsic_matrix, image, frame_size, pred_pose=None):
#     h, w = frame_size

#     def process_pose_to_joints(pose):
#         sk = Skeleton(pose)
#         joints_2d = torch.tensor(sk.get_2d_points(
#             w,
#             h,
#             extrinsic_matrix=extrinsic_matrix,
#             intrinsic_matrix=intrinsic_matrix,
#         ))
#         joints = torch.stack([joints_2d[:, 0], joints_2d[:, 1]], 1)
#         return joints

#     gt_joints = process_pose_to_joints(gt_pose)
#     if pred_pose is not None:
#         pred_joints = process_pose_to_joints(pred_pose)
#         plot_skeleton_2d(image, gt_joints=gt_joints, pred_joints=pred_joints)
#     else:
#         plot_skeleton_2d(image, gt_joints)


def batch_show(imgs, sub_titles=None, title=None, row_labels=None,
               col_labels=None, cmap='gray', vrange_mode='fixed',
               ret_fig=False, font_size=(20, 20, 20),
               font_type='Times New Roman', sub_size=(3, 3)):
    """ Show images. 
    Args:
        imgs: Supposed to be an 2-d list or tuple. Each element is an image in numpy.ndarray format.
        sub_titles: Titles of each subplot.
        title: The image overall title.
        cmap: When the image only has two dimension, or only select one band, the cmap used by
            matplotlib.pyplot. Default is gray.
        vrange_mode: When the input image is monochrome, whether use a cmap value range auto min-max,
            or use a fixed range from 0 to 255. Select from ('auto', 'fixed').
        ret_fig: Whether return the processed input image.
        font_size: tuple/list/int/float, the font sizes of row, column, and subtitle. If input type is
            int/float, set all font sizes the same.
        font_type: str, the font name of your desired font type.
    """
    if not (isinstance(imgs[0], list) or isinstance(imgs[0], tuple)):
        imgs = [imgs]
    if not (isinstance(font_size, list) or isinstance(font_size, tuple)):
        font_size = (font_size, font_size, font_size)
    rows = len(imgs)
    cols = max([len(i) for i in imgs])

    # plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(sub_size[0]*cols, sub_size[1]*rows), sharey=True)
    if rows == 1:
        axs = [axs]
    if cols == 1:
        axs = [[i] for i in axs]
    axs = np.array(axs)
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            img = imgs[i][j]
            if sub_titles is not None and len(sub_titles) > i and len(sub_titles[i]) > j:
                sub_title = sub_titles[i][j]
            else:
                sub_title = ''
            if len(img.shape) == 2 or img.shape[0] == 1 or img.shape[-1] == 1:
                if vrange_mode == 'fixed':
                    axs[i, j].imshow(img, cmap=cmap, vmin=0, vmax=255)
                else:
                    axs[i, j].imshow(img, cmap=cmap)
            else:
                axs[i, j].imshow(img)
            axs[i, j].set(xticks=[], yticks=[])
            if row_labels is not None and len(row_labels) > i:
                axs[i, j].set_ylabel(row_labels[i], fontsize=font_size[0], fontname=font_type)
            if col_labels is not None and len(col_labels) > j:
                axs[i, j].set_xlabel(col_labels[j], fontsize=font_size[1], fontname=font_type)
            if sub_title != '':
                axs[i, j].set_title(sub_title, fontsize=font_size[2], y=-0.15, fontname=font_type)

    for ax in axs.flat:
        ax.label_outer()

    if title is not None:
        fig.suptitle(title, fontsize=30)
    plt.tight_layout()

    if ret_fig:
        return fig
