from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import DataLoader, Dataset

from utils.utils import read_json, read_jsonl, swap_sample


class PairwiseDataset(Dataset):
    """Pairwise preference rows: Text, Output1, Output2, Better, optional Image, ID."""

    def __init__(
        self,
        data_path: str,
        processing_func: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        swap: bool = False,
        data_start_idx: Optional[int] = None,
        data_end_idx: Optional[int] = None,
        order_ids_path: Optional[str] = None,
        strict_order: bool = True,
    ):
        self.items: List[Dict[str, Any]] = []
        if data_path.endswith(".jsonl"):
            self.items = read_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported data_path extension: {data_path}")

        if processing_func:
            self.items = processing_func(self.items)

        if data_start_idx is not None:
            self.items = self.items[data_start_idx:]
        if data_end_idx is not None:
            self.items = self.items[:data_end_idx]

        if swap:
            self.items = [swap_sample(item) for item in self.items]

        if order_ids_path is not None:
            order_ids = read_json(order_ids_path)
            id_to_item = {item["ID"]: item for item in self.items}

            missing = [i for i in order_ids if i not in id_to_item]
            if strict_order and missing:
                raise ValueError(
                    f"[order_ids_path] {len(missing)} IDs not found in dataset. e.g. {missing[:5]}"
                )

            ordered_items = [id_to_item[i] for i in order_ids if i in id_to_item]
            leftovers = [
                item for item in self.items if item["ID"] not in set(order_ids)
            ]
            ordered_items.extend(leftovers)

            if not ordered_items:
                raise ValueError("No items matched the provided ordered_ids.")
            self.items = ordered_items
            print(
                f"[ORDER] Loaded {len(self.items)} items in provided order. "
                f"First 5: {[x['ID'] for x in self.items[:5]]}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def _collate_fn(batch: Any) -> Any:
    if isinstance(batch, dict):
        first_val = next(iter(batch.values()))
        n = len(first_val)
        out: List[Dict[str, Any]] = []
        for i in range(n):
            sample = {k: v[i] for k, v in batch.items()}
            out.append(sample)
        return out
    return batch


def create_dataloader(
    dataset: PairwiseDataset, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
    )
