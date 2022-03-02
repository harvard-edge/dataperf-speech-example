from pathlib import Path
import fire
import yaml
from selection.selection import TrainingSetSelection
from selection.serialization import deserialize


def main(input_samples, outdir="/workdir"):
    sample_data = deserialize.deserialize_from_pb(input_samples)
    target_mswc_vectors = sample_data["target_mswc_vectors"]
    target_ids = sample_data["target_ids"]
    nontarget_mswc_vectors = sample_data["nontarget_mswc_vectors"]
    nontarget_ids = sample_data["nontarget_ids"]

    selection = TrainingSetSelection(
        target_vectors=target_mswc_vectors,
        target_ids=target_ids,
        nontarget_vectors=nontarget_mswc_vectors,
        nontarget_ids=nontarget_ids,
    )

    train = selection.select()

    assert Path(
        outdir
    ).is_dir(), (
        f"{outdir} does not exist, please specify --outdir as a command line argument"
    )
    output = Path(outdir) / "train.yaml"
    with open(output, "w") as fh:
        yaml.dump(
            dict(target_ids=train.target_ids, nontarget_ids=train.nontarget_ids),
            fh,
            default_flow_style=None,
            sort_keys=False,
        )


if __name__ == "__main__":
    fire.Fire(main)
