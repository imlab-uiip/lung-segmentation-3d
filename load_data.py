import numpy as np
import nibabel as nib

def loadDataGeneral(df, path, append_coord):
    """
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.
    
    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """
    X, y = [], []
    for i, item in df.iterrows():
        img = nib.load(path + item[0]).get_data()
        mask = nib.load(path + item[1]).get_data()
        mask = np.clip(mask, 0, 255)
        cmask = (mask * 1. / 255)
        out = cmask
        X.append(img)
        y.append(out)
    X = np.expand_dims(X, -1)
    y = np.expand_dims(y, -1)
    y = np.concatenate((1 - y, y), -1)
    y = np.array(y)
    # Option to append coordinates as additional channels
    if append_coord:
        n = X.shape[0]
        inpShape = X.shape[1:]
        xx = np.empty(inpShape)
        for i in xrange(inpShape[1]):
            xx[:, i, :, 0] = i
        yy = np.empty(inpShape)
        for i in xrange(inpShape[0]):
            yy[i, :, :, 0] = i
        zz = np.empty(inpShape)
        for i in xrange(inpShape[2]):
            zz[:, :, i, 0] = i
        X = np.concatenate([X, np.array([xx] * n), np.array([yy] * n), np.array([zz] * n)], -1)

    print '### Dataset loaded'
    print '\t{}'.format(path)
    print '\t{}\t{}'.format(X.shape, y.shape)
    print '\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max())
    return X, y
