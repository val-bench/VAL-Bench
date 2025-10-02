from datasets import load_dataset, load_from_disk


def load_from_path(dataset_path, token=None, split='train'):
    if dataset_path.startswith("s3://") or dataset_path.startswith("local"):
        dataset = load_from_disk(dataset_path)
    elif len(dataset_path.split("/")) == 2:
        dataset = load_dataset(dataset_path, token=token)[split]
    else:
        raise ValueError(f"Invalid dataset path format: {dataset_path}")

    return dataset