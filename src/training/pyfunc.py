"""MLflow pyfunc wrapper so the registered model can be served with ``mlflow models serve``."""

from __future__ import annotations

import pickle

import mlflow.pyfunc
import pandas as pd


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as fh:
            self.model = pickle.load(fh)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        k = int((params or {}).get("k", 10))
        recs = self.model.recommend(model_input["user_id"].astype(int).to_numpy(), k)
        return pd.DataFrame({"user_id": list(recs), "recommended_item_ids": [list(map(int, v)) for v in recs.values()]})
