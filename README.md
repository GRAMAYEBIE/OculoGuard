👁️ OculoGuard v2.0: Dual-Expert Ocular Diagnostic Ecosystem
OculoGuard v2.0 is an end-to-end MLOps framework designed for the automated, resilient, and interpretable screening of Diabetic Retinopathy (DR) and Glaucoma. Unlike monolithic models, OculoGuard uses a dual-expert architecture supported by a distributed streaming pipeline.

🚀 Key Features
Dual-Expert Engine: * DR Expert: EfficientNet-B3 (TensorFlow) for vascular pathology classification.

Glaucoma Expert: ResNet-50 (PyTorch) for precise Optic Disc segmentation and CDR (Cup-to-Disc Ratio) regression.

Asynchronous Pipeline: Leveraging Apache Kafka to handle high-throughput image ingestion and inference, ensuring system zero-downtime.

Explainable AI (XAI): Integrated Grad-CAM heatmaps to provide clinicians with visual evidence for every diagnostic decision.

Metadata Audit Trail: A persistent PostgreSQL layer recording every transaction (Image URI, Prediction, Confidence) for medical-legal traceability.

Interactive UI: Built with Streamlit for real-time clinical uploads and reporting.

🛠️ System Architecture

graph LR
A[User/Clinic] -->|Upload Image| B(Streamlit UI)
B -->|Produce Message| C(Apache Kafka)
C -->|Consume & Infer| D{Dual Expert Engine}
D -->|DR Class| E[EfficientNet-B3]
D -->|Glaucoma CDR| F[ResNet-50]
E & F -->|Store Results| G[(PostgreSQL Audit Trail)]
E & F -->|Generate XAI| H[Grad-CAM Heatmaps]
H -->|Return| B

📊 Performance & MonitoringHandling Class Imbalance: 
Implementation of Class Weights and SMOTE to address the scarcity of severe pathological cases ($<0.172\%$ in raw data)
Optimization: Use of ReduceLROnPlateau, EarlyStopping, and ModelCheckpoint for robust convergence.
Monitoring: Integrated tracking of model drift and system latency via MLflow.
📁 Repository Structureglaucoma_segmentation.ipynb: 
PyTorch implementation for Optic Disc analysis.
pprojet.ipynb: TensorFlow implementation for DR classification and XAI.
app/: Streamlit frontend and API logic.
pipeline/: Kafka producer/consumer scripts and Database schemas.
🔧 Installation & SetupClone the repo:
Clone the repo
git clone https://github.com/YourUsername/OculoGuard-v2.git
Install dependencies:

Bash
pip install -r requirements.txt
Start Infrastructure:

Launch your Zookeeper & Kafka brokers.

Initialize the PostgreSQL database.

Run the App:

Bash
streamlit run app.py

🤝 Contributing
Contributions are welcome! If you want to improve the segmentation masks or add new experts (e.g., Macular Degeneration), please open an issue or a PR.

Developed by Meschac Ayebie Data Scientist | Data Engineer | MLOps Enthusiast

