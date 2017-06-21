from load_data import loadDataGeneral
from build_model import build_model
import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = '/path/to/dataset/idx-train.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)

    # Load training data
    append_coords = True
    X, y = loadDataGeneral(df, path, append_coords)

    # Build model
    inp_shape = X[0].shape
    model = build_model(inp_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Visualize model
    plot_model(model, 'model.png', show_shapes=True)

    model.summary()

    ##########################################################################################
    checkpointer = ModelCheckpoint('model.{epoch:03d}.hdf5', period=5)

    model.fit(X, y, batch_size=1, epochs=50, callbacks=[checkpointer], validation_split=0.2)
