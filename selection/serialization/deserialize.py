import numpy as np
from selection.serialization.serialization import protoc_pb2

def deserialize_from_pb(filepath):
    samples_pb = protoc_pb2.Samples()
    with open(filepath, "rb") as fh:
        samples_pb.ParseFromString(fh.read())

    target_mswc_vectors = []
    target_ids = []
    nontarget_mswc_vectors = []
    nontarget_ids = []
    for sample in samples_pb.samples:
        if sample.sample_type == protoc_pb2.SampleType.TARGET:
            target_mswc_vectors.append(sample.mswc_embedding_vector)
            target_ids.append(sample.sample_id)
        elif sample.sample_type == protoc_pb2.SampleType.NONTARGET:
            nontarget_mswc_vectors.append(sample.mswc_embedding_vector)
            nontarget_ids.append(sample.sample_id)
        else:
            raise ValueError("unsupported type for deserialization")
    return dict(
        target_mswc_vectors=np.array(target_mswc_vectors),
        target_ids=target_ids,
        nontarget_mswc_vectors=np.array(nontarget_mswc_vectors),
        nontarget_ids=nontarget_ids,
    )
