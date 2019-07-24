import json
import h5py


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    for fname, recons in reconstructions.items():
        #print(fname.type, reconstructions.items().type)
        #print(str(out_dir) +'/' +str(fname))
        with h5py.File(str(out_dir) +'/' +str(fname), 'w') as f:
            f.create_dataset('reconstruction', data=recons)
