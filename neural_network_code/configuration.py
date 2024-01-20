from dataclasses import dataclass, field, asdict


@dataclass
class Configuration:
    mu: float
    manual_seed: int = field(metadata={'json_key': 'manualSeed'})
    training_iterations: int = field(metadata={'json_key': 'trainingIterations'})
    group_size: int = field(metadata={'json_key': 'groupSize'})
    test_perc: float = field(metadata={'json_key': 'testPerc'})
    data_path: str = field(metadata={'json_key': 'dataPath'})
    ch_type: str = field(metadata={'json_key': 'chType'})
    batch_size: int = field(metadata={'json_key': 'batchSize'})
    snr_value: int = field(metadata={'json_key': 'snrValue'})
    node_counts: list = field(metadata={'json_key': 'nodeCounts'})

    @classmethod
    def from_dict(cls, data):
        return cls(
            mu=data.get('mu'),
            manual_seed=data.get('manualSeed'),
            training_iterations=data.get('trainingIterations'),
            group_size=data.get('groupSize'),
            test_perc=data.get('testPerc'),
            data_path=data.get('dataPath'),
            ch_type=data.get('chType'),
            snr_value=data.get('snrValue'),
            node_counts=data.get('nodeCounts'),
            batch_size=data.get('batchSize')
        )


if __name__ == '__main__':
    pass
