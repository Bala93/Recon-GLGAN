import h5py


def save_reconstructions(reconstructions, out_dir):

    for fname, recons in reconstructions.items():

        with h5py.File(str(out_dir) +'/' +str(fname), 'w') as f:
            f.create_dataset('reconstruction', data=recons)
