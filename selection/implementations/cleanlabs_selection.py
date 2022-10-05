from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import tqdm

# include additional dependencies as needed:
import cleanlab
from cleanlab.classification import CleanLearning
from sklearn_extra.cluster import KMedoids

from selection.selection import TrainingSetSelection, TrainingSet


class CleanlabsSelection(TrainingSetSelection):
    def __init__(self, **kwargs) -> None:
        super(CleanlabsSelection, self).__init__(**kwargs)

    def select(self):
        """"
        Returns: 
            TrainingSet
        """

        if self.audio_flag:
            print(self.embeddings["nontargets"][0]["audio"])

        target_to_classid = {
            target: ix + 1
            for ix, target in enumerate(self.embeddings["targets"].keys())
        }
        target_to_classid["nontarget"] = 0

        per_class_size = self.train_set_size // len(target_to_classid.keys())

        target_samples = np.array(
            [
                sample["feature_vector"]
                for target, samples in self.embeddings["targets"].items()
                for sample in samples
            ]
        )

        target_labels = np.array(
            [
                target_to_classid[target]
                for (target, samples) in self.embeddings["targets"].items()
                for sample in samples
            ]
        )
        # this maps each row in target_samples back to the sample ID
        reverse_map_target_index_to_sample_id = [
            (target, sample["ID"])
            for (target, samples) in self.embeddings["targets"].items()
            for sample in samples
        ]

        nontarget_samples = np.array(
            [sample["feature_vector"] for sample in self.embeddings["nontargets"]]
        )
        nontarget_labels = np.zeros(nontarget_samples.shape[0])

        
        print("Using cleanlab and then K-medoids clustering to select data subset ...")
        SEED = 123
        NUM_FOLDS = 10  # can significantly speed up code by decreasing this to 2
        SUBSET_SIZE = 120
        CLEANLAB_FRAC = 0.5  # what fraction of datapoints to delete with lowest cleanlab scores (beyond default cleanlab filter)
        clf = sklearn.ensemble.VotingClassifier(
                    estimators=[
                        ("svm", sklearn.svm.SVC(probability=True)),
                        ("lr", sklearn.linear_model.LogisticRegression()),
                    ],
                    voting="soft",
                    weights=None,
                )
        cl = cleanlab.classification.CleanLearning(clf, seed=SEED, verbose=True, cv_n_folds=NUM_FOLDS)
        Xs = np.vstack(
                    [
                        target_samples,
                        nontarget_samples,
                    ]
                )
        ys = np.concatenate(
                    [
                        target_labels,
                        nontarget_labels,
                    ]
                )
        ys = ys.astype(int)
        issues = cl.find_label_issues(Xs, ys)
        bad_indices_filter = np.where(issues['is_label_issue'])[0]
        num_delete_cleanlab = int(CLEANLAB_FRAC * len(ys))
        bad_indices_rank = issues['label_quality'].values.argsort()[:num_delete_cleanlab]
        bad_indices = np.unique(np.concatenate((bad_indices_filter,bad_indices_rank),0))
        issues.drop(bad_indices, axis=0, inplace=True)
        inds_to_keep = issues.index.values

        # Cluster remaining data:
        big_labels = ys[inds_to_keep,np.newaxis] * 1000  # multiply labels by large value to ensure clustering is label-aware:
        data_to_cluster = np.concatenate([Xs[inds_to_keep], big_labels], axis=1)
        # clstr = sklearn.cluster.KMeans(n_clusters=SUBSET_SIZE)
        # cluster_ids = clstr.fit_predict(data_to_cluster)
        clstr = KMedoids(n_clusters=SUBSET_SIZE, init="k-medoids++")
        clstr.fit(data_to_cluster)
        coreset_inds = clstr.medoid_indices_
        
        # Extract indices:
        best_indices = inds_to_keep[coreset_inds]
        best_target_train_ixs = [x for x in best_indices if x < len(target_samples)]
        best_nontarget_train_ixs = [x-len(target_samples) for x in best_indices if x >= len(target_samples)]

        selected_targets = {k: [] for k in self.embeddings["targets"].keys()}
        for target_ix in best_target_train_ixs:
            target, clip_id = reverse_map_target_index_to_sample_id[target_ix]
            selected_targets[target].append(clip_id)

        selected_nontargets = [
            self.embeddings["nontargets"][sample_ix]["ID"]
            for sample_ix in best_nontarget_train_ixs
        ]

        training_set = TrainingSet(
            targets=selected_targets, nontargets=selected_nontargets,
        )

        return training_set
