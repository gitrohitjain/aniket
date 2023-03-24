git clone https://github.com/ultralytics/yolov5
python -m venv myenv
source myenv/bin/activate
pip install -r yolov5/requirements.txt
python detect.py --weights <model_path/best.pt> --img 640 --source <img_path.png> --conf 0.24 --max-det 1 --save-crop

#### output image crop will be saved in yolov5/runs/detect/exp/crops/solution