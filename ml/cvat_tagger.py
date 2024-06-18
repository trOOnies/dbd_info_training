import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Optional, Dict
from shutil import rmtree, move
from zipfile import ZipFile
from xmltodict import parse

NONE_STR = "NONE"

LabelId = int
LabelName = str
LabelRef = Dict[LabelId, LabelName]
CVATLabelId = int
CVATIdRef = Dict[LabelName, CVATLabelId]


class CVATTagger:
    def __init__(self) -> None:
        self.data: Optional[pd.DataFrame] = None
        self.label_ref: Optional[LabelRef] = None
        self.token: Optional[str] = None
        self.cvat_id_ref: Optional[CVATIdRef] = None

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

    def load_data(
        self,
        path: str,
        label_ref: LabelRef,
        id_col: str = "cvat_id"
    ) -> None:
        self.data = pd.read_csv(path)
        self.data = self.data.set_index(id_col, verify_integrity=True)
        self.label_ref = label_ref

    def load_preds(self, preds: np.ndarray) -> None:
        self.data["label_pred"] = preds  # TODO: Add probas option

    def commit(self) -> None:
        self.data["label_id"] = self.data["label_pred"]
        self.data = self.data.drop("label_pred", axis=1)

    # * LOGIN

    def login(self) -> None:
        auth = requests.post(
            f"http://{os.environ['CVAT_HOST']}/api/auth/login",
            json={
                "email": os.environ["CVAT_MAIL"],
                "password": os.environ["CVAT_PASSWORD"]
            }
        )
        assert auth.ok, str(auth.json())
        auth = auth.json()  # {"key": "string"}
        self.token = auth["key"]

    def _get_headers(self) -> dict:
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.cvat+json',
            'Authorization': f"Token {self.token}"
        }

    def logout(self) -> None:
        auth = requests.get(
            f"http://{os.environ['CVAT_HOST']}/api/auth/logout",
            headers={'Authorization': f"Token {self.token}"}
        )
        assert auth.ok, str(auth.json())

    # * TAGGING

    def get_tasks(self) -> dict:
        tasks = requests.get(f"http://{os.environ['CVAT_HOST']}/api/tasks", headers=self._get_headers())
        assert tasks.ok, str(tasks.json())
        return tasks.json()

    def download_task(
        self,
        id: int,
        path_out: str
    ) -> None:
        assert path_out.endswith(".xml")
        zip_path = f"{path_out[:-4]}.zip"

        requests.get(
            f"http://{os.environ['CVAT_HOST']}/api/tasks/{id}/dataset?format=CVAT+for+images+1.1&scheme=json",
            headers={'Authorization': f"Token {self.token}"}
        )
        task = requests.get(
            f"http://{os.environ['CVAT_HOST']}/api/tasks/{id}/dataset?format=CVAT+for+images+1.1&scheme=json&action=download",
            headers={'Authorization': f"Token {self.token}"}
        )  # * 2 times so that it actually downloads
        with open(zip_path, "wb") as f:
            f.write(task.content)

        fd = os.path.dirname(path_out)
        temp_fd = os.path.join(fd, f"{os.path.basename(path_out)[:-4]}_TEMP_FD")
        os.mkdir(temp_fd)

        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_fd)
        os.remove(zip_path)

        move(
            os.path.join(temp_fd, "annotations.xml"),
            path_out
        )
        rmtree(temp_fd)

    def get_task_info(self, path: str):
        with open(path, "r") as f:
            task_info = f.read()

        task_info = parse(task_info)
        task_info = task_info["annotations"]

        task_info = [
            {
                (k[1:] if k[0] == "@" else k): v
                for k, v in img.items() if k in {"@id", "@name", "tag"}
            }
            for img in task_info["image"]
        ]
        task_info = [
            {
                "id": int(img["id"]),
                "name": img["name"],
                "tag": (img["tag"]["@label"] if "tag" in img else None),
                "is_manual": (img["tag"]["@source"] == "manual" if "tag" in img else True)
            }
            for img in task_info
        ]
        task_info = pd.DataFrame(task_info)
        task_info = task_info.rename({"id": "frame_id"}, axis=1)
        task_info = task_info.set_index("frame_id", verify_integrity=True)

        return task_info

    def get_label_count(self) -> int:
        total_labels = requests.get(
            f"http://{os.environ['CVAT_HOST']}/api/labels",
            headers=self._get_headers()
        )
        assert total_labels.ok
        return total_labels.json()["count"]

    def load_cvat_id_ref(self) -> None:
        """Label example:
        {'id': 1, 'name': "A Nurse's Calling", 'color': '#40C04F', 'attributes': [],
        'type': 'tag', 'sublabels': [], 'project_id': 1, 'parent_id': None, 'has_parent': False}
        """
        self.cvat_id_ref = {}
        next_page = f"http://{os.environ['CVAT_HOST']}/api/labels?page=1"
        while next_page is not None:
            res = requests.get(next_page, headers=self._get_headers()).json()
            self.cvat_id_ref.update({r["name"]: r["id"] for r in res["results"]})
            next_page = res["next"]

    def load_preds(self, preds_path: str) -> None:
        with open(preds_path) as f:
            preds = f.readlines()
        preds = [v[:-1] if v[-1] == "\n" else v for v in preds]
        self.data["label_pred"] = np.array(preds, dtype=int)

    def process_preds(self) -> None:
        self.data["pred_cvat_id"] = [
            self.cvat_id_ref[self.label_ref[v]]
            for v in self.data["label_pred"]
        ]  # LabelId -> LabelName -> CVATLabelId

    def send_preds(self, job_id: int) -> None:
        # * Only upload non-null preds labels
        ser = self.data["pred_cvat_id"].copy()
        ser = ser[ser.notnull()]
        ser = ser.astype(int)

        data = {
            "tags": [
                {
                    "frame": frame,
                    "label_id": lid
                } for frame, lid in ser.items()
            ]
        }

        ann = requests.patch(
            f"http://{os.environ['CVAT_HOST']}/api/jobs/{job_id}/annotations?action=update",
            headers=self._get_headers(),
            json=data
        )
        assert ann.ok, str(ann.json())

    def delete_annotations(self, job_id: int) -> None:
        ann = requests.delete(
            f"http://{os.environ['CVAT_HOST']}/api/jobs/{job_id}/annotations",
            headers=self._get_headers()
        )
        assert ann.ok, str(ann.json())

    # url = f"http://{os.environ['CVAT_HOST']}/api/jobs"
    # response = requests.get(url)

    # # url = f"{os.environ['CVAT_HOST']}/api/tasks/{TASK_ID}/annotations"
    # # headers = {
    # #     'Content-Type': 'application/json',
    # #     'Accept': 'application/json',
    # # }
    # # data = ""
    # # response = requests.put(url, headers=headers, json=data)

    # if response.status_code == 200:
    #     print("Annotations updated successfully")
    # else:
    #     print(f"Failed to update annotations: {response.content}")

    # * PLOTTING

    def _create_ax(
        self,
        size: float,
        sp: int,
        max_in_row: int
    ) -> tuple:
        if sp == 1:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        else:
            if sp <= max_in_row:
                fig, ax = plt.subplots(1, sp, figsize=(sp * size, 1.0 * size))
            else:
                rows = (sp // max_in_row) + int(sp % max_in_row > 0)
                fig, ax = plt.subplots(rows, max_in_row, figsize=(max_in_row * size, rows * size))

        fig.patch.set_visible(False)
        if isinstance(ax, np.ndarray):
            if ax.ndim == 2:
                for ax_i in ax:
                    for ax_j in ax_i:
                        ax_j.axis("off")
            else:
                for ax_i in ax:
                    ax_i.axis("off")
        else:
            ax.axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)
        return fig, ax

    def _add_image(self, ax, id: int) -> None:
        ax.text(0.5, 1.1, f"IID={id}", fontdict={"color": "white"}, ha='center', va='center', transform=ax.transAxes)
        image = mpimg.imread(
            os.path.join(os.environ["CROPS_FD"], os.environ["SELECTED_FD"], self.data["name"].at[id])
        )
        ax.imshow(image, extent=[0, 50, 0, 50])

    def plot_img(self, id: int, size: float = 1.00) -> None:
        fig, ax = self._create_ax(size, sp=1, max_in_row=1)
        self._add_image(ax, id)
        fig.show()

    def plot_preds(
        self,
        label_id: int,
        size: float = 1.00,
        max_in_row: int = 20
    ) -> None:
        vals = self.data["name"][self.data["label_pred"] == label_id].index.values
        if vals.size == 0:
            return

        fig, ax = self._create_ax(size, sp=vals.size, max_in_row=max_in_row)

        if isinstance(ax, np.ndarray):
            if ax.ndim == 2:
                for pos, i in enumerate(vals):
                    self._add_image(ax[pos // max_in_row][pos % max_in_row], i)
            else:
                for pos, i in enumerate(vals):
                    self._add_image(ax[pos], i)
        else:
            self._add_image(ax, vals[0])

        fig.show()
