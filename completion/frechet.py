"""
Calculates Fréchet Point Cloud Distance

****************************************************************************
    Since FPCD compares the latent space of a separate model, this file
    requires a pretrained PointNet classifier from the alpha-davidson
    downstream-benchmarks repository. As a reult, this file also needs
    to be run in that virtual environment, not the virtual environment
    used to train a SnowflakeNet model
****************************************************************************

Author: Ben Wagner
Date Created: 27 Feb 2025
Date Edited:  03 Mar 2025
"""

import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import sys
sys.path.append("../../voxel_data/")
sys.path.append("../")
from models.pointnet import create_pointnet_model


def train_model(train_feats_path, train_labels_path, val_feats_path,
                val_labels_path, model_save_path, latent_save_path):
    '''
    '''

    train_feats = np.load(train_feats_path)
    train_labels = np.load(train_labels_path)
    train_ds = tf.data.Dataset.from_tensor_slices((train_feats, train_labels))
    train_ds = train_ds.batch(batch_size=32, drop_remainder=True)

    val_feats = np.load(val_feats_path)
    val_labels = np.load(val_labels_path)
    val_ds = tf.data.Dataset.from_tensor_slices((val_feats, val_labels))
    val_ds = val_ds.batch(batch_size=32, drop_remainder=True)

    nPoints = train_feats.shape[1]
    nFeats = train_feats.shape[2]
    nClasses = len(np.unique(train_labels))

    model, latent = create_pointnet_model(
        num_points=nPoints,
        num_features=nFeats,
        num_classes=nClasses,
        is_regression=False,
        is_pointwise_prediction=False
    )

    model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        metrics=["sparse_categorical_accuracy"]
    )

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                     patience=7, verbose=1)
    history = model.fit(train_ds, validation_data=val_ds, epochs=100,
                        verbose=0, callbacks=[best_model_callback, reduce_lr])
    
    latent.save(filepath=latent_save_path)
    
    return model, latent


def load_model(model_path, latent_path):
    '''
    Loads in pretrained models

    Parameters:
        model_path: str - path to complete model
        latent_path: str - path to model that gives latent space

    Returns:
        model: tf.keras.Model - complete model
        latent: tf.keras.Model - latent space
    '''

    return tf.keras.models.load_model(model_path), tf.keras.models.load_model(latent_path)


def compute_cov_mean(feats):
    '''
    Computes covarience and mean of latent vector

    Parameters:
        feats: np.ndarray - latent vector

    Returns:
        cov: np.ndarray - covarience of latent vector
        mean: np.ndarray - mean of latent vector
    '''
    cov, mean = np.cov(feats, rowvar=False), np.mean(feats, axis=0)
    return cov, mean


def frechet(pred, gt):
    '''
    Computes Fréchet Point Cloud Distance

    Parameters:
        pred: np.ndarray - latent vector of predicted cloud
        gt: np.ndarray - latent vector of ground truth cloud

    Returns:
        fpcd: float - Fréchet Point Cloud Distance between pred and gt
    '''
    cov_p, mean_p = compute_cov_mean(pred)
    cov_r, mean_r = compute_cov_mean(gt)

    diff = mean_r - mean_p
    covmean = sqrtm(cov_r @ cov_p)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fpcd = np.sum(diff ** 2) + np.trace(cov_r + cov_p - 2 * covmean)
    return fpcd


def visualize(frechet_results, f_score):
    '''
    
    '''

    model_names = frechet_results['model']
    dists = frechet_results['fpcd']

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f"Fréchet Point Cloud Distance Comparison")
    ax.set_title(f"Model F1-Score: {f_score}")
    ax.set_ylabel("FPCD")
    ax.set_xlabel("Model")
    ax.bar(model_names, dists, label=model_names)
    plt.savefig("frechet_results.png")


def main(gt_path, preds_path, model_path, latent_path, train_feats_path,
         train_labels_path, val_feats_path, val_labels_path, test_feats_path,
         test_labels_path, new=True):
    '''
    '''

    if new:
        model, latent = train_model(train_feats_path, train_labels_path,
                                    val_feats_path, val_labels_path,
                                    model_path, latent_path)
    else:
        model, latent = load_model(model_path, latent_path)

    y_pred = model.predict(np.load(test_feats_path))
    model_f_score = f1_score(np.load(test_labels_path), np.argmax(y_pred, axis=-1), average='weighted')

    gt_feats = latent.predict(np.load(gt_path))
    frechet_results = np.ndarray((len(preds_path.keys())),
                                 dtype=[('model', 'object'), ('fpcd', np.float32)])

    for i, k in enumerate(preds_path.keys()):

        pred = latent.predict(np.load(preds_path[k]))
        frechet_results[i] = (k, frechet(pred, gt_feats))

    visualize(frechet_results, model_f_score)
    



if __name__ == '__main__':

    GT_PATH = 'justMg_gts.npy'
    PRED_PATHS = {
        '22Mg Only' : 'justMg_preds.npy',
        '22Mg and 16O' : 'MgO_preds.npy'
        }
    
    NEW_TRAIN = False
    MODEL_PATH = './exp/checkpoints/frechet/model.keras'
    LATENT_PATH = './exp/checkpoints/frechet/latent.keras'
    TRAIN_F_PATH = '../../voxel_data/Mg22_data/Mg22_size2048_train_features.npy'
    TRAIN_L_PATH = '../../voxel_data/Mg22_data/Mg22_size2048_train_labels.npy'
    VAL_F_PATH = '../../voxel_data/Mg22_data/Mg22_size2048_val_features.npy'
    VAL_L_PATH = '../../voxel_data/Mg22_data/Mg22_size2048_val_labels.npy'
    TEST_F_PATH = '../../voxel_data/Mg22_data/Mg22_size2048_test_features.npy'
    TEST_L_PATH = '../../voxel_data/Mg22_data/Mg22_size2048_test_labels.npy'
    
    main(GT_PATH, PRED_PATHS, MODEL_PATH, LATENT_PATH, TRAIN_F_PATH,
         TRAIN_L_PATH, VAL_F_PATH, VAL_L_PATH, TEST_F_PATH,
         TEST_L_PATH, new=NEW_TRAIN)
    
    print("Done")