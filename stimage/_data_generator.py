import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.preprocessing import image
from _imgaug import seq_aug
from sklearn.neighbors import KDTree


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, adata, dim=(299, 299), n_channels=3, genes=None, aug=False, tile_path="tile_path"):
        'Initialization'
        self.dim = dim
        self.adata = adata
        self.n_channels = n_channels
        self.genes = genes
        self.num_genes = len(genes)
        self.aug = aug
        self.tile_path = tile_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.adata.n_obs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of IDs
        obs_temp = self.adata.obs_names[index]

        # Generate data
        X_img = self._load_img(obs_temp)
        y = self._load_label(obs_temp)

        return X_img, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.adata.n_obs)

    def _load_img(self, obs):
        img_path = self.adata.obs.loc[obs, 'tile_path']
        X_img = image.load_img(img_path, target_size=self.dim)
        X_img = image.img_to_array(X_img).astype('uint8')
        #         X_img = np.expand_dims(X_img, axis=0)
        #         n_rotate = np.random.randint(0, 4)
        #         X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
        if self.aug:
            X_img = seq_aug(image=X_img)
#         X_img = preprocess_resnet(X_img)
        return X_img

    def _load_label(self, obs):
        batch_adata = self.adata[obs, self.genes].copy()

        return tuple([batch_adata.to_df()[i].values for i in self.genes])

    def get_classes(self):
        return self.adata.to_df().loc[:, self.genes]


class DataGenerator_LSTM_one_output(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, adata, dim=(299, 299), n_channels=3, genes=None, aug=False):
        'Initialization'
        self.dim = dim
        self.adata = adata
        self.n_channels = n_channels
        self.genes = genes
        self.num_genes = len(genes)
        self.aug = aug
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.adata.n_obs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of IDs
        obs_temp = self.adata.obs_names[index]

        # Generate data
        X_img = self._load_img(obs_temp)
        y = self._load_label(obs_temp)
        
        # Generate neighbour data
        candidates = self.adata.obs[['imagecol', 'imagerow']]
        quary_spots = np.array(candidates.loc[obs_temp,:]).reshape(1, -1)
        nearest_index = self.get_nearest_index(quary_spots, candidates, k_nn=7, leaf_size=self.adata.n_obs//2)
        np.random.shuffle(nearest_index.flat)
        obs_temp_neighbour = self.adata.obs_names[nearest_index.ravel()]
        
        X_img_neighbour = np.stack([self._load_img(x) for x in obs_temp_neighbour], axis=0)
        X_img = np.expand_dims(X_img, axis=0)
        X_img_final = np.concatenate([X_img_neighbour, X_img], axis=0)

        return X_img_final, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.adata.n_obs)

    def _load_img(self, obs):
        img_path = self.adata.obs.loc[obs, 'tile_path']
        X_img = image.load_img(img_path, target_size=self.dim)
        X_img = image.img_to_array(X_img).astype('uint8')
        #         X_img = np.expand_dims(X_img, axis=0)
        #         n_rotate = np.random.randint(0, 4)
        #         X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
        if self.aug:
            X_img = seq_aug(image=X_img)
#         X_img = preprocess_resnet(X_img)
#         X_img = np.expand_dims(X_img, axis=0)
        return X_img

    def _load_label(self, obs):
        batch_adata = self.adata[obs, self.genes].copy()

        return tuple([batch_adata.to_df()[i].values for i in self.genes])

    def get_classes(self):
        return self.adata.to_df().loc[:, self.genes]
    
    def get_nearest_index(self, quary_spots, candidates, k_nn, leaf_size):
        tree = KDTree(candidates, leaf_size=leaf_size)
        _, indices = tree.query(quary_spots, k=k_nn)
        return indices.ravel()[1:]


class DataGenerator_LSTM_multi_output(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, adata, dim=(299, 299), n_channels=3, genes=None, aug=False, tile_path="tile_path"):
        'Initialization'
        self.dim = dim
        self.adata = adata
        self.n_channels = n_channels
        self.genes = genes
        self.num_genes = len(genes)
        self.aug = aug
        self.tile_path = tile_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.adata.n_obs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of IDs
        obs_temp = self.adata.obs_names[index]
        
        # Generate neighbour data
        candidates = self.adata.obs[['imagecol', 'imagerow']]
        quary_spots = np.array(candidates.loc[obs_temp,:]).reshape(1, -1)
        nearest_index = self.get_nearest_index(quary_spots, candidates, k_nn=7, leaf_size=self.adata.n_obs//2)
        np.random.shuffle(nearest_index.flat)
        obs_temp_neighbour = self.adata.obs_names[nearest_index.ravel()]
        
        # Generate data
        X_img_final = self._load_neighbour(self._load_img, obs_temp, obs_temp_neighbour)
        y_final = self._load_neighbour(self._load_label, obs_temp, obs_temp_neighbour)

        return X_img_final, y_final
    
    def _load_neighbour(self, fn, obs_temp, obs_temp_neighbour):
        obs_data = fn(obs_temp)
        obs_neighbour = np.stack([fn(x) for x in obs_temp_neighbour], axis=0)
        obs_data = np.expand_dims(obs_data, axis=0)
        obs_data_final = np.concatenate([obs_neighbour, obs_data], axis=0)
        return obs_data_final
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.adata.n_obs)

    def _load_img(self, obs):
        img_path = self.adata.obs.loc[obs, self.tile_path]
        X_img = image.load_img(img_path, target_size=self.dim)
        X_img = image.img_to_array(X_img).astype('uint8')
        #         X_img = np.expand_dims(X_img, axis=0)
        #         n_rotate = np.random.randint(0, 4)
        #         X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
        if self.aug:
            X_img = seq_aug(image=X_img)
#         X_img = preprocess_resnet(X_img)
#         X_img = np.expand_dims(X_img, axis=0)
        return X_img

    def _load_label(self, obs):
        batch_adata = self.adata[obs, self.genes].copy()

        return np.array(tuple([batch_adata.to_df()[i].values for i in self.genes]), dtype = "f"+",f"*(self.num_genes - 1))

    def get_classes(self):
        return self.adata.to_df().loc[:, self.genes]
    
    def get_nearest_index(self, quary_spots, candidates, k_nn, leaf_size):
        tree = KDTree(candidates, leaf_size=leaf_size)
        _, indices = tree.query(quary_spots, k=k_nn)
        return indices.ravel()[1:]