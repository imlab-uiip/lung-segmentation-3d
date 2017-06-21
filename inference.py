from load_data import loadDataGeneral

import numpy as np
import pandas as pd
import nibabel as nib
from keras.models import load_model

from scipy.misc import imresize
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure

def IoU(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def saggital(img):
    """Extracts midle layer in saggital axis and rotates it appropriately."""
    return img[:, img.shape[1] / 2, ::-1].T

img_size = 128

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = 'Demo/idx-val.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)

    # Load test data
    append_coords = False
    X, y = loadDataGeneral(df, path, append_coords)

    n_test = X.shape[0]
    inpShape = X.shape[1:]

    # Load model
    model_name = 'trained_model.hdf5' # Model should be trained with the same `append_coords`
    model = load_model(model_name)

    # Predict on test data
    pred = model.predict(X, batch_size=1)[..., 1]

    # Compute scores and visualize
    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    for i in range(n_test):
        gt = y[i, :, :, :, 1] > 0.5 # ground truth binary mask
        pr = pred[i] > 0.5 # binary prediction
        # Save 3D images with binary masks if needed
        if False:
            tImg = nib.load(path + df.ix[i].path)
            nib.save(nib.Nifti1Image(255 * pr.astype('float'), affine=tImg.get_affine()), df.ix[i].path+'-pred.nii.gz')
            nib.save(nib.Nifti1Image(255 * gt.astype('float'), affine=tImg.get_affine()), df.ix[i].path + '-gt.nii.gz')
        # Compute scores
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print df.ix[i]['path'], ious[i], dices[i]

        # Rescaling images to be within [0, 1].
        t_img = exposure.rescale_intensity(nib.load(path + df.ix[i]['path']).get_data(), out_range=(0, 1))
        # Creating 3x4 table previews
        lungs = np.zeros((img_size * 3, img_size * 4)) # Slices from original grayscale image
        mask = np.zeros((img_size * 3, img_size * 4)) # Slices from predicted mask
        gt_mask = np.zeros((img_size * 3, img_size * 4)) # Slices from ground truth mask
        # Fill [0, 0] cell with saggital view of lungs
        lungs[:img_size, :img_size] = imresize(saggital(t_img), [img_size, img_size]) * 1. / 256
        mask[:img_size, :img_size][imresize(saggital(pred[i]), [img_size, img_size]) > 128] = 1
        gt_mask[:img_size, :img_size][imresize(saggital(y[i][..., 1]), [img_size, img_size]) > 128] = 1
        # Fill the rest of the cells with 11 slices in z direction
        for k in range(1, 12):
            yy, xx = k / 4, k % 4 # Cell coordinates
            zz = int(t_img.shape[-1] * (k * 1. / 12)) # z coordinate of a slice
            lungs[yy * img_size: (yy + 1) * img_size, xx * img_size: (xx + 1) * img_size] = t_img[:, :, -zz]
            mask[yy * img_size: (yy + 1) * img_size, xx * img_size: (xx + 1) * img_size][pr[:, :, -zz]] = 1
            gt_mask[yy * img_size: (yy + 1) * img_size, xx * img_size: (xx + 1) * img_size][gt[:, :, -zz]] = 1
        # Combining masks to get a pretty picture
        prv = rgb2hsv(gray2rgb(lungs))
        mask_hsv = rgb2hsv(np.dstack([gt_mask, np.zeros_like(mask), mask]))
        prv[..., 0] = mask_hsv[..., 0]
        prv[..., 1] = mask_hsv[..., 1] * 0.9

        io.imsave('Demo/Predictions/' + df.ix[i]['path'] + '-preview.png', hsv2rgb(prv))
        io.imsave('Demo/Predictions/' + df.ix[i]['path'] + '-mask.png', np.dstack([gt_mask, mask, mask]))


    print 'Mean IoU:'
    print ious.mean()

    print 'Mean Dice:'
    print dices.mean()
