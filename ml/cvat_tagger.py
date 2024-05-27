import pandas as pd
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from numpy import ndarray


class CVATTagger:
    def __init__(self) -> None:
        self.data: Optional[pd.DataFrame] = None
        self.label_ref: Optional[dict] = None

    @property
    def data_loaded(self) -> bool:
        return self.data is not None

    @property
    def preds_loaded(self) -> bool:
        return self.data_loaded and "label_pred" in self.data.columns

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"CVATTagger(data={self.data_loaded}, preds={self.preds_loaded})"

    def load_data(self, path: str, label_ref: dict) -> None:
        self.data = pd.read_csv(path)
        self.label_ref = label_ref

    def load_preds(self, preds: "ndarray") -> None:
        self.data["label_pred"] = preds  # TODO: Add probas option

    def commit(self) -> None:
        self.data["label_id"] = self.data["label_pred"]
        self.data = self.data.drop("label_pred", axis=1)

    def tag(self) -> None:
        # TODO
        ...
