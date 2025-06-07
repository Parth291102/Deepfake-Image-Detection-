# 🧠 AI vs. AI: Deep Learning for DeepFake Image Detection

Detecting DeepFakes with deep learning models — from simple CNNs to state-of-the-art Dual-Stream EfficientNet architectures.

## 👨‍💻 Team Members

* Parth Ripal Parikh (200597041)
* Astha Sanjaykumar Bhalodiya (200575709)
* Mrudani Vishal Hada (200597322)

---

## 🎯 Project Motivation

With the rise of DeepFake technology, it's becoming increasingly difficult to distinguish between real and manipulated images. This poses significant risks in journalism, security, and public trust. Our goal is to develop a robust system that can detect DeepFake images using deep learning techniques.

---

## 🧾 Problem Statement

Develop a deep learning model to classify images as **real** or **fake**, leveraging different architectures including:

* Simple CNN
* VGG16 (Transfer Learning)
* Dual-Stream Custom CNN
* Dual-Stream EfficientNet

---

## 📁 Dataset

* **Source**: [Flickr-Faces-HQ (FFHQ)](https://www.kaggle.com/datasets/parthripalbhai/neural-network-stylegan-real-fake-dataset)
* **Real Images**: 7,763 high-quality, 512×512 PNG face images.
* **Fake Images**: 10,000 synthetic faces generated using [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada).
* **Augmentation**: Applied using Keras `ImageDataGenerator`.

---

## 🔬 Exploratory Data Analysis (EDA)

* Class distribution check
* Color channel and brightness intensity analysis
* Frequency spectrum and texture (contrast, energy, homogeneity)
* Mislabel and quality verification

---

## 🏗️ Models Overview

### 1. 📦 Simple CNN

* Accuracy: 60%
* AUC: 0.61
* Weak generalization and misclassifications on real images.

### 2. 🧠 VGG16 (Transfer Learning)

* Accuracy: 72%
* AUC: 0.79
* Better generalization and balance between precision & recall.

### 3. 🔄 Dual-Stream Custom CNN

* Two parallel convolutional streams: Spatial & Frequency
* Accuracy: 97%
* AUC: 0.99

### 4. ⚡ Dual-Stream EfficientNet (Best)

* Accuracy: **98%**
* AUC: **1.00**
* State-of-the-art classification with near-perfect precision/recall.

---

## 🧠 Dual-Stream Architecture

> Two input streams improve detection:

* **Spatial Stream**: Captures texture, artifacts.
* **Frequency Stream**: Detects subtle manipulation patterns (via DFT).
* **Fusion**: Feature concatenation and joint classification.

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* ROC Curve & AUC
* Confusion Matrix
* Misclassification Insight

---

## 🔧 Optimization Techniques

* Real-time data augmentation
* EarlyStopping & ModelCheckpoint
* L2 Regularization & Dropout
* Adam Optimizer with custom LR schedule

---

## 📊 Comparative Analysis

| Model                       | Accuracy | AUC      | Comments                                |
| --------------------------- | -------- | -------- | --------------------------------------- |
| Simple CNN                  | 60%      | 0.61     | High bias, poor generalization          |
| VGG16                       | 72%      | 0.79     | Better via transfer learning            |
| Dual-Stream Custom CNN      | 97%      | 0.99     | Excellent dual-feature performance      |
| 🚀 Dual-Stream EfficientNet | **98%**  | **1.00** | Best results with robust generalization |

---

## 📚 References

1. [FaceForensics++](https://arxiv.org/abs/1901.08971)
2. [Face Warping Artifacts](https://doi.org/10.1109/CVPRW.2019.00149)
3. [DeepFake Detection using VGGNet](https://doi.org/10.1109/CVPR42600.2020.00923)
4. [Kaggle Dataset](https://www.kaggle.com/datasets/parthripalbhai/neural-network-stylegan-real-fake-dataset)
5. [Kaggle: Simple CNN](https://www.kaggle.com/code/parthripalbhai/neural-network-project-cnn)
6. [Kaggle: VGGNet Model](https://www.kaggle.com/code/parthripalbhai/neural-network-vggnet)
7. [Kaggle: Dual-Stream Custom CNN](https://www.kaggle.com/code/mrudanihada/nn-ds-customcnn)
8. [Kaggle: Dual-Stream EfficientNet](https://www.kaggle.com/code/parthripalbhai/neural-network-dual-stream-efficientnet-cnn)

---

## 📥 How to Run

1. Clone this repo:

   ```bash
   git clone https://github.com/yourusername/deepfake-image-detection.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:

   ```bash
   python train_model.py --model efficientnet
   ```
4. Evaluate:

   ```bash
   python evaluate.py
   ```

---

## 🏁 Conclusion

Our results show that dual-stream models, especially with EfficientNet backbones, significantly improve DeepFake detection accuracy, precision, and robustness. This approach generalizes well across real-world datasets and manipulation techniques.

---

Would you like me to generate the full repo structure including a `requirements.txt`, training scripts template, and folder layout as well?
