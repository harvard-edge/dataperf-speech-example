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


class CleanlabSelection(TrainingSetSelection):
    def __init__(self, **kwargs) -> None:
        super(CleanlabSelection, self).__init__(**kwargs)

    def select(self):
        """"
        Returns: 
            TrainingSet
        """
        # Hyperparameters of this approach:
        NUM_FOLDS = 10  # can significantly speed up code by decreasing this to 2
        EXTRA_FRAC = 0.5  # what fraction of datapoints to delete with lowest cleanlab label quality scores (beyond default cleanlab filter)
        TARGET_FRAC = 0.5 # what fraction of total samples should be targets (vs nontargets)


        SUBSET_SIZE = self.train_set_size
        SEED = self.random_seed

        clf = sklearn.ensemble.VotingClassifier(
                    estimators=[
                        ("svm", sklearn.svm.SVC(probability=True)),
                        # ("lr", sklearn.linear_model.LogisticRegression()),
                    ],
                    voting="soft",
                    weights=None,
                )

        if self.audio_flag:
            print(self.embeddings["nontargets"][0]["audio"])

        target_to_classid = {
            target: ix + 1
            for ix, target in enumerate(self.embeddings["targets"].keys())
        }
        target_to_classid["nontarget"] = 0

        num_classes = len(self.embeddings['targets'].keys()) + 1 #num targets + one non-target class


        num_targets = int(self.train_set_size * TARGET_FRAC)
        per_target_class_size = num_targets // (len(target_to_classid.keys()) - 1)
        nontarget_class_size = int(self.train_set_size * (1 - TARGET_FRAC))
        print(f"num_targets: {num_targets}")
        print(f"per_target_class_size: {per_target_class_size}")
        print(f"nontarget_class_size: {nontarget_class_size}")
        

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

        
        print("Using cleanlab + K-medoids clustering to select data subset ...")
        cl = cleanlab.classification.CleanLearning(
            clf, seed=SEED, verbose=True, cv_n_folds=NUM_FOLDS,
            label_quality_scores_kwargs={"adjust_pred_probs": True}
        )
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
        if EXTRA_FRAC > 0:
            num_delete_cleanlab = int(EXTRA_FRAC * len(ys))
            bad_indices_rank = issues['label_quality'].values.argsort()[:num_delete_cleanlab]
            bad_indices = np.unique(np.concatenate((bad_indices_filter,bad_indices_rank),0))
        else:
            bad_indices = bad_indices_filter
        issues.drop(bad_indices, axis=0, inplace=True)
        inds_to_keep = issues.index.values

        # Cluster remaining data:
        og_ys = ys
        ys = ys[inds_to_keep]
        Xs = Xs[inds_to_keep]
        print("Class distribution: ")
        unique, counts = np.unique(ys, return_counts=True)
        print(np.asarray((unique, counts)).T)
        print("Finding coresets among remaining clean data ...")
        best_indices = np.array([], dtype=int)
        for y in np.unique(ys):
            print(f"Finding coreset for class {y} ...")
            per_class_size = per_target_class_size if y > 0 else nontarget_class_size
            y_inds_to_keep = np.where(ys == y)[0]
            Xs_y = Xs[y_inds_to_keep]
            # can try Kmeans instead also:
            # clstr = sklearn.cluster.KMeans(n_clusters=SUBSET_SIZE)
            # cluster_ids = clstr.fit_predict(data_to_cluster)
            clstr = KMedoids(n_clusters=per_class_size, init="k-medoids++")
            clstr.fit(Xs_y)
            coreset_y_inds = clstr.medoid_indices_
            best_indices = np.concatenate((best_indices, inds_to_keep[y_inds_to_keep[coreset_y_inds]]))
        
        print("Class distribution post-selection: ")
        print(best_indices)
        unique, counts = np.unique(og_ys[best_indices], return_counts=True)
        print(np.asarray((unique, counts)).T)
        print(len(best_indices))

        # Extract indices:
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
