<div align="center">

# 🚀 Nexis 😷🖥️
### **⚡ AI-Powered Real-Time Face Mask Detection System ⚡**  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-FF5733?style=for-the-badge&logo=yolov5&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-FC4F30?style=for-the-badge&logo=matplotlib&logoColor=white)

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![AI](https://img.shields.io/badge/AI-Computer%20Vision-blueviolet?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YOLO-orange?style=flat-square)

</div>

---

## 🎯 About

**Nexis** is an AI-powered system designed to **detect face mask usage in real-time** 🎥 using images or video streams. Traditional CCTV systems cannot automatically monitor mask compliance, making manual surveillance inefficient and error-prone. Nexis solves this problem by combining **computer vision** 👁️, **image preprocessing** 🖼️, and **YOLO object detection** 🤖 to monitor public health safety efficiently.

🏥 The system is suitable for deployment in **bus stops, hospitals, schools, malls**, and other public spaces. It can detect people **without masks** ❌ or **wearing masks incorrectly** ⚠️, and generate **alerts or compliance reports** 📊.

---

## 🧠 Proposed Methodology

### 1️⃣ Data Acquisition 📦
- 🌐 Public face mask datasets from **[Roboflow](https://roboflow.com)** and **[Kaggle](https://www.kaggle.com)**  
- 🏷️ Datasets include **bounding box annotations** and images labeled with/without masks  

### 2️⃣ Image Preprocessing 🎨
- 📝 Use XML annotations to identify face regions  
- ⚙️ Preprocess images: **resize**, **normalize**, and prepare for training  

### 3️⃣ Mask Detection Pipeline 🔍
- 🎯 Apply **YOLOv8** to detect faces and classify as **masked** ✅ or **unmasked** ❌  
- 👥 Supports **multiple faces per frame** for video or images  

### 4️⃣ Tools & Libraries 🛠️
- 🐍 **Python** – programming language  
- 🤖 **YOLO** – object detection model  
- 📷 **OpenCV** – image/video processing  
- 🔢 **NumPy** – numerical operations  
- 📈 **Matplotlib** – visualization of results

---

## 📥 Installation

### 1️⃣ Clone Repository 📋
```bash
git clone https://github.com/Dilhara11/Nexis.git
cd nexis
```
### 2️⃣ Create Virtual Environment 🌐
```bash
python -m venv venv
```
### 3️⃣ Activate Virtual Environment ⚡
```bash
Windows

venv\Scripts\activate

Linux/Mac

source venv/bin/activate
```
### 4️⃣ Install Dependencies 📦
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure and File Guide (For New Users 👨‍💻👩‍💻)

This section explains what each folder/file does so a new user can quickly understand how to use this project after cloning. ⬇️

### 📄 Root Files

- 📖 `README.md` - Project overview and setup instructions
- 📋 `requirements.txt` - Python dependencies to install
- 🧠 `yolov8n.pt` - Base YOLO model weights
- 🔖 `yolo26n.pt` - Additional model weights/checkpoint used in experiments

### 🗂️ Main Folders

- 🎨 `preprocessing/` - Scripts for preparing and cleaning dataset inputs
- 🏋️ `trainig/` - Model training script (folder name is kept as-is in this project)
- 🔮 `inference/` - Run detection on images using trained weights
- 📊 `evaluation/` - Evaluate model performance
- 💾 `runs/` - Auto-generated YOLO outputs (training results, checkpoints, validation artifacts)

### 📝 File-by-File Usage

#### 🎨 `preprocessing/`

- 🧹 `data_cleaning.py` - Cleans raw annotation/image data
- ⚙️ `process_dataset.py` - Prepares dataset into training-ready format
- 🗑️ `remove_duplicates.py` - Removes duplicate entries/images
- 👀 `visualize_annotations.py` - Visual check of annotation correctness

### 🔬 Digital Image Processing in `process_dataset.py`

💡 `process_dataset.py` applies a compact enhancement pipeline before training. The goal is to produce cleaner and more consistent face images so YOLO learns stronger mask/non-mask features under real-world lighting and camera noise.

#### 🌊 Technique 1: Gaussian Blur (Noise Reduction)

- 🔍 **What it is:** A smoothing filter using a Gaussian kernel (`cv2.GaussianBlur(img, (5,5), 0)`).
- 💡 **Why we use it:** Reduces high-frequency sensor noise and tiny artifacts that can distract detection models.
- ✅ **Benefit for training:** Improves feature stability so the model focuses on meaningful facial/mask patterns instead of random pixel noise.

#### 🎨 Technique 2: Color Space Conversion to LAB

- 🔍 **What it is:** Converts BGR image to LAB color space (`cv2.cvtColor(img, cv2.COLOR_BGR2LAB)`).
- 💡 **Why we use it:** LAB separates luminance (`L`) from color channels (`A`, `B`), so we can enhance brightness/contrast without distorting colors.
- ✅ **Benefit for training:** Makes preprocessing more lighting-aware and consistent across bright and dim scenes.

#### ✨ Technique 3: CLAHE on Luminance Channel

- 🔍 **What it is:** Contrast Limited Adaptive Histogram Equalization on `L` channel (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`).
- 💡 **Why we use it:** Increases local contrast in low-visibility regions while limiting over-amplification of noise.
- ✅ **Benefit for training:** Recovers details on faces and masks in shadows, backlight, or uneven illumination.

#### 🔄 Technique 4: Merge and Convert Back to BGR

- 🔍 **What it is:** Recombines processed `L` with original `A/B`, then converts LAB back to BGR.
- 💡 **Why we use it:** Keeps output compatible with standard OpenCV/YOLO data pipelines.
- ✅ **Benefit for training:** Maintains color realism while still applying robust contrast enhancement.

#### 🔐 Dataset Integrity Steps in the Script

- 🎯 **Split-aware processing:** Applies preprocessing separately to `train`, `valid`, and `test`.
- 🏷️ **Label preservation:** Copies YOLO label files unchanged to keep annotation alignment intact.
- ⚙️ **Config-based paths:** Uses `.env` variables (`DATASET_PATH`, `PROCESSED_DATASET_PATH`) for flexible local setup.

#### 🎯 Why This Pipeline Matters

🚀 This preprocessing stack improves generalization by reducing noise sensitivity and improving visibility consistency. In practical terms, it helps the detector remain reliable across different cameras, weather/lighting conditions, and image quality levels.

#### 🏋️ `trainig/`

- 🔥 `train_YOLO.py` - Main training script to train the face-mask detector

#### 🔮 `inference/`

- 🎯 `detect_image.py` - Runs object detection on an image and outputs predictions

#### 📊 `evaluation/`

- 📈 `evaluate_model.py` - Evaluates trained model metrics/results

#### 💾 `runs/detect/`

🔧 These are generated during training/validation and should be treated as outputs:

- 🏃 `train/`, `train2/`, `train3/`, ... - Different training runs
- 📝 `args.yaml` - Parameters used for that run
- 📊 `results.csv` - Per-epoch training metrics (when available)
- 🏆 `weights/best.pt` - Best checkpoint from a run
- 💾 `weights/last.pt` - Last checkpoint from a run
- ✅ `val/`, `val2/`, ... - Validation outputs

---

## 🚀 Quick Start Workflow for New Cloners

#### 1️⃣ Clone and set up environment 📦

```bash
git clone https://github.com/Dilhara11/Nexis.git
cd Nexis
python -m venv venv
```

#### 2️⃣ Activate virtual environment ⚡

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3️⃣ Install dependencies 📥

```bash
pip install -r requirements.txt
```

#### 4️⃣ (Optional) Run preprocessing scripts 🎨

If you are preparing your own dataset, run preprocessing scripts.

#### 5️⃣ Train model 🏋️

```bash
python ./trainig/train_YOLO.py
```

#### 6️⃣ Run inference 🔮

```bash
python ./inference/detect_image.py
```

#### 7️⃣ Evaluate model 📊

```bash
python ./evaluation/evaluate_model.py
```

---

## 💡 Notes for First-Time Users

- 🔒 **Keep your trained checkpoints safe** (`runs/detect/*/weights/`).
- 🏆 **Use `best.pt` for inference** if you want best validation performance from a run.
- 🗑️ **The `runs/` directory can grow quickly**; clean old runs if disk usage becomes high.

---

<div align="center">

### 🌟 Made with ❤️ for Public Health Safety 🌟

**[⬆ Back to Top](#-nexis-️)**

</div>