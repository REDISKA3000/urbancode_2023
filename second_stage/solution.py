from typing import Dict
from collections import Counter
from abc import ABC, abstractmethod

from typing import List
import math
import yaml
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path


from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO



class FacadeModel(ABC):
    def __init__(self):
        self._points = []

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass

    def todict(self) -> Dict:
        self._validate()
        d = dict()
        for x, y, n, k in self._points:
            d[str((int(x), int(y)))] = [int(n), int(k)]
        return d

    def _validate(self):
        for point in self._points:
            x, y, n, k = point
            assert all(v >= 0 for v in point), "All values must be non-negative"
            assert n >= 1, "Number of the floor must be greater than zero"
            assert k >= 1, "Number of the vertical must be greater than zero"
            assert isinstance(n, int), "Number of the floor must be integer"
            assert isinstance(k, int), "Number of the vertical must be integer"
        counter = Counter((n, k) for _, _, n, k in self._points)
        assert all(occur == 1 for occur in counter.values()), "(n, k) pairs must be unique"


class DummyFacadeModel(FacadeModel):
    def __init__(self):
        super().__init__()

    def build(self, boxes: np.ndarray) -> None:
        x_centers = boxes[:,0]
        y_centers = boxes[:,1]

        xcs_sorted = np.sort(x_centers)
        eps_x = 2.5 * abs(np.mean((np.roll(xcs_sorted, 1) - xcs_sorted)[1:-2]))
        ycs_sorted = np.sort(y_centers)
        eps_y = 2.5 * abs(np.mean((np.roll(ycs_sorted, 1) - ycs_sorted)[1:-2]))

        if math.isnan(eps_x): eps_x = 0.1
        if math.isnan(eps_y): eps_y = 0.1

        vertical_md = DBSCAN(eps=eps_x)
        verticals = vertical_md.fit_predict(x_centers.reshape(-1, 1)) + 1
        floor_md = DBSCAN(eps=eps_y)
        floors = floor_md.fit_predict(y_centers.reshape(-1, 1)) + 1

        points = np.dstack([x_centers, y_centers, floors, verticals])[0].astype(int)
        points = points[(points[:, 2] != 0) & (points[:, 3] != 0)]
        unique_indices = np.array(list({tuple(x): i for i, x in enumerate(points[:, [2, 3]])}.values()))
        if len(unique_indices):
            self._points = points[unique_indices].tolist()

def _glob_images(folder: Path, exts: List[str] = ('*.jpg', '*.png',)) -> List[Path]:
    images = []
    for ext in exts:
        images += list(folder.glob(ext))
    return images

model = YOLO('best-14.pt')
input_folder = './test/images'

output_dict = {}
input_folder = Path(input_folder)
images_path = _glob_images(input_folder)

for img_path in images_path:
    # img = np.asarray(Image.open(img_path))
    print(img_path)
    preds = model.predict(source = img_path, save = True, save_txt=True)
    boxes = np.array(preds[0].boxes.xywh.tolist())
    facade = DummyFacadeModel()
    facade.build(boxes)
    output_dict[img_path.stem] = facade.todict()

output_file = '/content/preds.yaml'
output_file = Path(output_file)

with output_file.open('w') as f:
      yaml.safe_dump(output_dict, f)

