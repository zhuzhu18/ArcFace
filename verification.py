import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


def calculate_roc(thresholds, embeddings, actual_issame, nrof_folds=10, pca=0):

    embeddings1 = embeddings[0::2].data.numpy()  # N * 512
    embeddings2 = embeddings[1::2].data.numpy()  # N * 512
    assert (embeddings1.shape[0] == embeddings2.shape[0]) and (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)    # 400
    k_fold = KFold(n_splits=nrof_folds, shuffle=True)

    tprs = np.zeros((nrof_folds, nrof_thresholds))     # 10 * 400
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))      # (10, )
    best_thresholds = np.zeros((nrof_folds))     # (10, )
    indices = np.arange(nrof_pairs)     # indices of pairs, (N, )

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), axis=1)    # distance, (N, )

    for fold_idx, (train_indices, test_indices) in enumerate(k_fold.split(indices)):
        # print(train_indices, test_indices)
        if pca > 0:
            # print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_indices]
            embed2_train = embeddings2[train_indices]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))    # (400, )
        for threshold_idx, threshold in enumerate(thresholds):
            tpr, fpr, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_indices], actual_issame[train_indices])
        best_threshold_index = np.argmax(acc_train)
        # print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]    # (10, ), 训练的那一折上的最佳阈值
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_indices],
                                                                                                 actual_issame[test_indices])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_indices],
                                                      actual_issame[test_indices])

    tpr = np.mean(tprs, 0)    # (400, ), 在每个阈值下的平均tpr(测试的那些折)
    fpr = np.mean(fprs, 0)    # (400, ), 在每个阈值下的平均fpr(测试的那些折)

    return tpr, fpr, accuracy, best_thresholds  # accuracy和best_thresholds分别为10折对应的测试精度和训练最佳阈值


def calculate_accuracy(threshold, dist, actual_issame):
    predict_mask = np.less(dist, threshold)       # 实际预测的正例
    tp = np.sum(np.logical_and(predict_mask, actual_issame))
    fp = np.sum(np.logical_and(predict_mask, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_mask), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc
