# Nexis 😷🖥️
**AI-Powered Real-Time Face Mask Detection System**  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-FF5733?style=for-the-badge&logo=yolov5&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-FC4F30?style=for-the-badge&logo=matplotlib&logoColor=white)

---

## About

**Nexis** is an AI-powered system designed to **detect face mask usage in real-time** using images or video streams. Traditional CCTV systems cannot automatically monitor mask compliance, making manual surveillance inefficient and error-prone. Nexis solves this problem by combining **computer vision**, **image preprocessing**, and **YOLO object detection** to monitor public health safety efficiently.

The system is suitable for deployment in **bus stops, hospitals, schools, malls**, and other public spaces. It can detect people **without masks** or **wearing masks incorrectly**, and generate **alerts or compliance reports**.

---

## Proposed Methodology

### 1️⃣ Data Acquisition
- Public face mask datasets from **[Roboflow](https://roboflow.com)** and **[Kaggle](https://www.kaggle.com)**  
- Datasets include **bounding box annotations** and images labeled with/without masks  

### 2️⃣ Image Preprocessing
- Use XML annotations to identify face regions  
- Preprocess images: **resize**, **normalize**, and prepare for training  

### 3️⃣ Mask Detection Pipeline
- Apply **YOLOv8** to detect faces and classify as **masked** or **unmasked**  
- Supports **multiple faces per frame** for video or images  

### 4️⃣ Tools & Libraries
- **Python** – programming language  
- **YOLO** – object detection model  
- **OpenCV** – image/video processing  
- **NumPy** – numerical operations  
- **Matplotlib** – visualization of results  

---

## Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Dilhara11/Nexis.git
cd nexis
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```
### 3️⃣ Activate Virtual Environment
```bash
Windows

venv\Scripts\activate

Linux/Mac

source venv/bin/activate
```
### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```