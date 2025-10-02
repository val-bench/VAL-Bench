def save_to_path(dataset, dataset_path, token=None, private=True):
    if dataset_path.startswith("s3://") or dataset_path.startswith("local"):
        dataset.save_to_disk(dataset_path)
    elif len(dataset_path.split("/")) == 2:
        dataset.push_to_hub(dataset_path, private=private, token=token)
    else:
        raise ValueError(f"Invalid dataset path format: {dataset_path}")
