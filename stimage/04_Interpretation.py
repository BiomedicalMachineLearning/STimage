from lime import lime_image
import skimage
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
import numpy as np
import scipy as sp
from scipy import ndimage as ndi
from skimage.morphology import area_opening
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, SGDRegressor
import joblib


# Clustering followed by Classification
def clustering(train_adata, path):
    clustering = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
    model_c = LogisticRegression(max_iter=10000,penalty='elasticnet',C=0.1,solver='saga',l1_ratio=0.5)
    clf_can_v_non_can = model_c.fit(train_adata.obsm["true_gene_expression"],clustering.fit_predict(train_adata.obsm["true_gene_expression"]))
    joblib.dump(clf_can_v_non_can, path+'pickle/resnet_block1_log_scaled_relu_clustering_logistic.pkl')


def model_predict_gene_nb(gene, gene_list, model):
    i = gene_list.index(gene)
    from scipy.stats import nbinom
    def model_predict(x):
        test_predictions = model.predict(x)
        n = test_predictions[i][:, 0]
        p = test_predictions[i][:, 1]
        y_pred = nbinom.mean(n, p)
        return y_pred.reshape(-1,1)
    return model_predict


def model_predict_gene(gene, gene_list_2, resnet_model, clf_resnet):
    i = gene_list_2.index(gene)
    def combine_model_predict(tile1):
        feature1 = resnet_model.predict(tile1)
        prediction = clf_resnet.predict_proba(feature1)
        return prediction[i]
    return combine_model_predict


def watershed_segment(image):
    annotation_hed = rgb2hed(image)
    annotation_h = annotation_hed[:,:,0]
    annotation_h *= 255.0 / np.percentile(annotation_h, q=0.01)
    thresh = skimage.filters.threshold_otsu(annotation_h)*0.9
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        annotation_h < thresh
    )
    distance = ndi.distance_transform_edt(im_fgnd_mask)
    coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=im_fgnd_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(annotation_h, markers, mask=im_fgnd_mask)
    im_nuclei_seg_mask = area_opening(labels, area_threshold=64).astype(np.int)
    map_dic = dict(zip(np.unique(im_nuclei_seg_mask), np.arange(len(np.unique(im_nuclei_seg_mask)))))
    im_nuclei_seg_mask = np.vectorize(map_dic.get)(im_nuclei_seg_mask)
    return im_nuclei_seg_mask


def LIME_heatmap(image_path, explainer, gene, image_fun):
    image = np.asarray(image_fun.load_img(image_path))
    explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=2, hide_color=0, num_samples=1000, segmentation_fn=watershed_segment)
    temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=10000, hide_rest=True)
    dict_heatmap = dict(explanation.local_exp[1])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    return mask, heatmap


