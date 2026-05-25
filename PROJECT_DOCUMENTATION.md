# ABSTRACT

The **"AI Face Recognition and Monitoring System"** is a real-time, intelligent surveillance and attendance management solution developed using the Python programming language and modern machine learning libraries. In today's rapidly evolving digital landscape, security and efficient monitoring have become critical necessities for educational institutions, corporate offices, and public spaces. Traditional manual methods of attendance tracking — such as roll calls, sign-in registers, and manual logbooks — are not only time-consuming and error-prone, but also highly susceptible to proxy attendance and data manipulation. Similarly, conventional CCTV-based monitoring systems merely record footage without any intelligent analysis, requiring human operators to constantly watch the feeds.

This project addresses these pressing challenges by implementing a comprehensive face recognition–based system powered by computer vision and deep learning techniques. The system leverages **OpenCV** (Open Source Computer Vision Library) for real-time video capture, frame processing, and image pre-processing; the **face_recognition** library — built on top of **dlib's** state-of-the-art deep learning model that uses a Residual Neural Network (ResNet) with 29 convolutional layers — for highly accurate face detection and identification; and the **Flask** micro web framework for delivering a responsive, browser-based user interface for interaction and monitoring.

A key highlight of the system is its robust **two-step face registration and training workflow**. Rather than relying on a single photograph — which often leads to poor recognition accuracy under varying lighting conditions, angles, and expressions — the administrator captures **50 photographs** of each new user directly through the browser's live camera feed using the browser-native `getUserMedia` API. These photographs are saved to the server disk without immediately impacting the recognition model. The administrator can subsequently review all unregistered users in a dedicated "Train Model" panel and trigger model training with a single click. During training, the system processes all 50 stored photographs for each person, computes a set of **128-dimensional face encodings** (using dlib's deep metric learning approach), selects the top 5 most diverse encodings through a maximal diversity selection algorithm, and updates the recognition model. This multi-sample approach results in significantly higher accuracy than single-photo registration, as it captures the natural variance in a person's facial appearance.

The system incorporates an automated attendance logging mechanism that records attendance with timestamps the very first time a registered person is detected by the camera on any given day. It intelligently prevents duplicate entries and classifies attendance status as either "Present" or "Late" based on a configurable time threshold. For security monitoring, the system includes an unknown face detection and alerting module that automatically captures and saves snapshots of unrecognised individuals, employing a sophisticated multi-layer throttling mechanism (spatial-bucket cooldowns, encoding-based deduplication, and global rate limiting) to prevent an overwhelming number of false alerts.

**Key features** of the system include: real-time face detection and recognition from live camera feeds with bounding box overlays and confidence percentages; browser-native live camera preview during face registration (eliminating server-side camera conflicts); multi-photo capture with a real-time progress bar and thumbnail preview strip; a two-step training workflow with a dedicated Train Model panel; automated attendance logging with date filtering and CSV/Excel export; an interactive alerts page with filter buttons, hover zoom, full-screen lightbox modal, and image download capability; session-based admin authentication with SHA-256 password hashing; and comprehensive error handling with user-friendly flash messages.

The entire system is built on a **SQLite** database for lightweight, file-based data persistence and is designed to be cost-effective, easily deployable, and scalable. The project totals approximately **3,500 lines of code** across Python backend modules, HTML templates, CSS stylesheets, and client-side JavaScript.

**Keywords:** Face Recognition, Computer Vision, OpenCV, Deep Learning, dlib, Attendance System, Python, Flask, Real-time Monitoring, getUserMedia, Multi-sample Training, SQLite, Machine Learning, Biometric Authentication

---

<div style="page-break-after: always;"></div>

# TABLE OF CONTENTS

| Chapter      | Title                                           | Page No. |
| ------------ | ----------------------------------------------- | -------- |
|              | Certificate                                     | i        |
|              | Declaration                                     | ii       |
|              | Acknowledgement                                 | iii      |
|              | Abstract                                        | iv       |
|              | Table of Contents                               | v        |
|              | List of Figures                                 | vi       |
|              | List of Tables                                  | vii      |
| **1**  | **INTRODUCTION**                          |          |
|              | 1.1 Introduction                                |          |
|              | 1.2 Motivation                                  |          |
|              | 1.3 Problem Statement                           |          |
|              | 1.4 Objectives of the Project                   |          |
|              | 1.5 Scope of the Project                        |          |
|              | 1.6 Organization of the Report                  |          |
| **2**  | **LITERATURE REVIEW AND SYSTEM ANALYSIS** |          |
|              | 2.1 Literature Survey                           |          |
|              | 2.2 Existing System                             |          |
|              | 2.3 Proposed System                             |          |
|              | 2.4 Comparison: Existing vs. Proposed System    |          |
|              | 2.5 Feasibility Study                           |          |
| **3**  | **SYSTEM REQUIREMENTS**                   |          |
|              | 3.1 Hardware Requirements                       |          |
|              | 3.2 Software Requirements                       |          |
|              | 3.3 Technology Stack                            |          |
| **4**  | **SYSTEM DESIGN**                         |          |
|              | 4.1 System Architecture                         |          |
|              | 4.2 Data Flow Diagrams (DFD)                    |          |
|              | 4.3 Entity-Relationship (ER) Diagram            |          |
|              | 4.4 Use Case Diagram                            |          |
|              | 4.5 Module Description                          |          |
|              | 4.6 Database Design                             |          |
| **5**  | **IMPLEMENTATION**                        |          |
|              | 5.1 Implementation Overview                     |          |
|              | 5.2 Project Directory Structure                 |          |
|              | 5.3 Core Backend Modules                        |          |
|              | 5.4 Frontend Design and Templates               |          |
|              | 5.5 Face Recognition Pipeline                   |          |
|              | 5.6 Key Algorithms                              |          |
| **6**  | **TESTING**                               |          |
|              | 6.1 Testing Strategy                            |          |
|              | 6.2 Types of Testing Performed                  |          |
|              | 6.3 Test Cases and Results                      |          |
| **7**  | **SCREENSHOTS AND OUTPUT**                |          |
|              | 7.1 Login Page                                  |          |
|              | 7.2 Dashboard                                   |          |
|              | 7.3 Face Registration (Camera Capture)          |          |
|              | 7.4 Face Registration (File Upload)             |          |
|              | 7.5 Train Model Panel                           |          |
|              | 7.6 Live Feed with Recognition                  |          |
|              | 7.7 Attendance Records                          |          |
|              | 7.8 Unknown Face Alerts                         |          |
| **8**  | **CONCLUSION AND FUTURE SCOPE**           |          |
|              | 8.1 Conclusion                                  |          |
|              | 8.2 Limitations                                 |          |
|              | 8.3 Future Scope                                |          |
| **9**  | **REFERENCES AND BIBLIOGRAPHY**           |          |
| **10** | **APPENDIX**                              |          |
|              | 10.1 Source Code – Key Modules                 |          |
|              | 10.2 Installation Guide                         |          |
|              | 10.3 Glossary of Terms                          |          |

---

<div style="page-break-after: always;"></div>

# LIST OF FIGURES

| Figure No. | Title                                               |
| ---------- | --------------------------------------------------- |
| 1.1        | Traditional Attendance Methods vs. Face Recognition |
| 4.1        | System Architecture Diagram                         |
| 4.2        | Context-Level DFD (Level 0)                         |
| 4.3        | Level 1 Data Flow Diagram                           |
| 4.4        | Entity-Relationship (ER) Diagram                    |
| 4.5        | Use Case Diagram                                    |
| 5.1        | Project Directory Structure                         |
| 5.2        | Face Recognition Pipeline Flowchart                 |
| 5.3        | Two-Step Registration and Training Workflow         |
| 5.4        | Unknown Face Throttling Mechanism                   |
| 7.1        | Login Page Screenshot                               |
| 7.2        | Dashboard Screenshot                                |
| 7.3        | Face Registration – Camera Capture Screenshot      |
| 7.4        | Face Registration – File Upload Screenshot         |
| 7.5        | Train Model Panel Screenshot                        |
| 7.6        | Live Feed with Recognition Overlay Screenshot       |
| 7.7        | Attendance Records Page Screenshot                  |
| 7.8        | Unknown Face Alerts Page Screenshot                 |
| 7.9        | Alert Lightbox Modal Screenshot                     |
| 7.10       | CSV Export Sample                                   |

---

# LIST OF TABLES

| Table No. | Title                                           |
| --------- | ----------------------------------------------- |
| 2.1       | Comparison of Existing and Proposed Systems     |
| 3.1       | Hardware Requirements                           |
| 3.2       | Software Requirements                           |
| 3.3       | Technology Stack Summary                        |
| 4.1       | Users Table Schema                              |
| 4.2       | Attendance Table Schema                         |
| 4.3       | Unknown Faces Table Schema                      |
| 4.4       | Admin Table Schema                              |
| 5.1       | Configuration Parameters and Their Descriptions |
| 6.1       | Test Cases and Results                          |

---

<div style="page-break-after: always;"></div>

# CHAPTER 1: INTRODUCTION

## 1.1 Introduction

In the modern era of digital transformation, the convergence of Artificial Intelligence (AI), Machine Learning (ML), and Computer Vision has unlocked unprecedented possibilities for automating and enhancing processes that were traditionally manual, tedious, and error-prone. One such domain that stands to benefit immensely from these technological advancements is attendance management and security monitoring in educational institutions, corporate offices, government buildings, and public spaces.

Face recognition technology has emerged as one of the most natural, non-intrusive, and efficient biometric identification methods available today. Unlike other biometric systems such as fingerprint scanners or iris readers that require physical contact or close proximity to specialised hardware, face recognition can identify individuals passively from a standard camera feed without requiring any conscious action from the person being identified. This makes it particularly suitable for large-scale, real-time applications such as automated attendance tracking and surveillance monitoring.

The **"AI Face Recognition and Monitoring System"** presented in this project report is a comprehensive, web-based application that integrates real-time face detection, recognition, automated attendance logging, and unknown face alerting into a single cohesive platform. Built using the Python programming language and the Flask web framework, the system provides an intuitive, browser-based interface that allows administrators to register new users through a sophisticated multi-photo capture workflow, train a face recognition model, monitor live camera feeds with real-time recognition overlays, view and export attendance records, and review security alerts for unrecognised individuals.

The project is designed with a strong emphasis on accuracy, usability, and practical deployability. It employs the industry-standard `face_recognition` library — which is built upon dlib's deep learning model utilising a Residual Neural Network (ResNet) architecture — to achieve high recognition accuracy. The two-step registration and training workflow, where 50 photographs are captured per user and the most diverse encodings are selected, ensures robust performance across varying lighting conditions, facial expressions, and head orientations.

## 1.2 Motivation

The motivation behind developing this project stems from several real-world observations and challenges faced in educational and institutional settings:

**Manual Attendance is Inefficient:** In most educational institutions across India, attendance is still recorded manually by teachers calling out names from a register or by passing around an attendance sheet. This process typically consumes 5 to 10 minutes of valuable lecture time per session. For a college with multiple classes running simultaneously, this translates to a significant loss of productive academic hours over the course of a semester.

**Proxy Attendance is Rampant:** Manual attendance systems are inherently vulnerable to proxy attendance, where a student marks the presence of an absent classmate. This is a widespread problem in Indian colleges and undermines the integrity of attendance records, which are often used for internal assessment and examination eligibility.

**Traditional Surveillance is Passive:** Most educational institutions and offices have CCTV cameras installed, but these systems merely record footage. There is rarely any real-time analysis or alerting mechanism. A security breach or unauthorised entry may go unnoticed until someone manually reviews hours of recorded footage, which is impractical and inefficient.

**Need for Automation and Intelligence:** With the rapid advancement and increasing affordability of AI and computer vision technologies, there is a clear opportunity to automate attendance management and make surveillance systems intelligent — capable of identifying individuals in real time, logging their presence automatically, and alerting administrators to potential security threats.

**Cost-Effective Solutions for Institutions:** Many existing commercial face recognition systems are prohibitively expensive for educational institutions, particularly in the North-Eastern region of India. This project demonstrates that a highly functional and accurate system can be built using open-source libraries and commodity hardware (a standard laptop with a webcam), making it accessible and affordable for institutions with limited budgets.

## 1.3 Problem Statement

The existing manual systems for attendance tracking and security monitoring in educational institutions face the following critical problems:

1. **Time Consumption:** Manual roll calls during each lecture consume significant academic time. In a class of 60 students, the roll call alone can take 7–10 minutes, which when accumulated across multiple sessions per day and hundreds of working days per semester, results in a substantial loss of productive teaching and learning time.
2. **Inaccuracy and Human Error:** Manual attendance registers are susceptible to clerical errors, illegible handwriting, missed entries, and accidental duplicate markings. These inaccuracies complicate subsequent record-keeping and analysis.
3. **Proxy Attendance:** The manual system has no inherent mechanism to verify the actual physical presence of a student. Proxy attendance — where one student marks attendance for another — is a well-documented and pervasive problem that compromises the reliability of attendance data.
4. **Lack of Real-Time Monitoring:** Conventional CCTV-based surveillance systems are passive. They record footage but do not perform any real-time analysis. An unknown individual entering a restricted area may not be noticed until long after the incident has occurred.
5. **No Centralised Digital Records:** Paper-based attendance registers make it extremely difficult to generate consolidated reports, track attendance trends, identify chronically absent students, or export data for administrative purposes. Digitising these records post-hoc is laborious and error-prone.
6. **Scalability Limitations:** Manual systems do not scale. As the number of students or entry points increases, the demands on human resources grow proportionally, making the system increasingly expensive and impractical.

## 1.4 Objectives of the Project

The primary objectives of this project are:

1. **To develop a real-time face recognition system** that can detect and identify registered individuals from a live camera feed with high accuracy and low latency.
2. **To automate the attendance marking process** by recognising registered individuals through their facial features and automatically logging their attendance with timestamps, thereby eliminating the need for manual roll calls.
3. **To implement a robust face registration workflow** that captures multiple (50) photographs per user to build a reliable and diverse set of face encodings, resulting in significantly higher recognition accuracy compared to single-photo systems.
4. **To create a comprehensive web-based dashboard** that provides real-time statistics, recent activity logs, and quick-access controls for administrators.
5. **To develop an unknown face detection and alerting system** that automatically captures snapshots of unrecognised individuals, logs them with timestamps, and presents them in an interactive alert management interface.
6. **To provide data export capabilities** that allow administrators to export attendance records in CSV format for further analysis, reporting, and integration with existing administrative workflows.
7. **To design the system for cost-effectiveness and easy deployment** by using open-source libraries, a lightweight SQLite database, and standard hardware (laptop with webcam), making it accessible to institutions with limited technical infrastructure and budgets.

## 1.5 Scope of the Project

The scope of the **AI Face Recognition and Monitoring System** encompasses the following functional areas:

**Within Scope:**

- Registration of new users through two methods: live camera capture (50 photos) or single photo upload.
- Training of the face recognition model using stored photographs with automated encoding generation.
- Real-time face detection and recognition from a live camera feed with visual overlays (bounding boxes, names, and confidence percentages).
- Automated attendance marking with date and time stamps, with automatic "Present" or "Late" status assignment based on a configurable time threshold.
- Prevention of duplicate attendance entries for the same person on the same day.
- Detection and logging of unknown (unregistered) faces with intelligent throttling to prevent alert flooding.
- A web-based administrative dashboard with statistics, recent activity, and quick actions.
- Attendance record viewing with date-based filtering and CSV export functionality.
- An interactive alerts management page with filter buttons (All/Unreviewed/Reviewed), hover zoom, full-screen lightbox, bulk selection, and bulk delete capabilities.
- User management features including editing student information (name, email, roll number, department, semester) and permanent user deletion (including photos and encodings).
- Session-based admin authentication with SHA-256 password hashing.

**Outside Scope (Limitations):**

- The system currently supports a single camera feed at a time.
- Multi-user concurrent access (multiple administrators logged in simultaneously) is not specifically addressed.
- The system does not include facial liveness detection (anti-spoofing measures against printed photos or video replays).
- Integration with existing college ERP (Enterprise Resource Planning) systems or Learning Management Systems (LMS) is not implemented in this version.
- Mobile application development is not within the scope of this project.
- The system is designed for deployment on a local network and not as a cloud-hosted service.

## 1.6 Organization of the Report

This project report is organised into **ten chapters**, each covering a specific aspect of the project:

**Chapter 1 – Introduction:** Provides an overview of the project, its motivation, the problem statement, objectives, and the scope of the system.

**Chapter 2 – Literature Review and System Analysis:** Surveys existing research and technologies related to face recognition and attendance systems. Compares the existing manual system with the proposed automated system and assesses the feasibility of the project from technical, economic, and operational perspectives.

**Chapter 3 – System Requirements:** Details the hardware and software requirements necessary for developing, deploying, and running the system, along with a description of the complete technology stack.

**Chapter 4 – System Design:** Presents the architectural design of the system through system architecture diagrams, Data Flow Diagrams (DFDs), Entity-Relationship (ER) diagrams, Use Case diagrams, and detailed module descriptions with database schema.

**Chapter 5 – Implementation:** Provides an in-depth technical account of the implementation process, including the project directory structure, core backend modules, frontend templates, the face recognition pipeline, and key algorithms used in the system.

**Chapter 6 – Testing:** Describes the testing strategy, the types of testing performed (unit, integration, system, and user acceptance testing), and presents detailed test cases with their expected and actual results.

**Chapter 7 – Screenshots and Output:** Showcases the visual output of the system through annotated screenshots of all major pages and functionalities.

**Chapter 8 – Conclusion and Future Scope:** Summarises the achievements of the project, discusses its limitations, and outlines potential future enhancements and extensions.

**Chapter 9 – References and Bibliography:** Lists all the books, research papers, websites, and documentation resources consulted during the development of this project.

**Chapter 10 – Appendix:** Contains supplementary material including key source code modules, the installation guide, and a glossary of technical terms.

---

<div style="page-break-after: always;"></div>

# CHAPTER 2: LITERATURE REVIEW AND SYSTEM ANALYSIS

## 2.1 Literature Survey

Face recognition has been an active area of research in computer vision and pattern recognition for over four decades. The evolution of face recognition algorithms can be broadly categorised into three generations:

**First Generation – Geometric Feature-Based Methods (1970s–1990s):** The earliest face recognition systems relied on measuring geometric features of the face — such as the distance between the eyes, the width of the nose, the length of the jaw, and the positions of the mouth and eyebrows. These measurements were compared against a stored template to determine identity. While conceptually simple, these methods were highly sensitive to variations in pose, expression, and imaging conditions, resulting in poor accuracy in real-world applications.

**Second Generation – Appearance-Based Methods (1990s–2010s):** This generation introduced statistical approaches that treated the entire face image (or significant portions of it) as a high-dimensional vector and employed dimensionality reduction and machine learning techniques for classification. Key algorithms from this era include:

- **Eigenfaces (PCA – Principal Component Analysis, 1991):** Proposed by Turk and Pentland, this method projects face images into a lower-dimensional "face space" defined by the eigenvectors (eigenfaces) of the covariance matrix of the training images. Recognition is performed by comparing the projection coefficients. While computationally efficient, Eigenfaces are sensitive to lighting changes and require all images to be frontal.
- **Fisherfaces (LDA – Linear Discriminant Analysis, 1997):** Fisher proposed an improvement over PCA by finding a projection that maximises the ratio of between-class scatter to within-class scatter, resulting in better discrimination between different individuals.
- **Local Binary Patterns Histograms (LBPH, 2006):** This approach divides the face into small regions, computes a Local Binary Pattern for each pixel by comparing it with its neighbours, and builds a histogram of the patterns for each region. The concatenated histograms form the face descriptor. LBPH is relatively robust to lighting changes and is computationally lightweight.

**Third Generation – Deep Learning–Based Methods (2014–Present):** The advent of deep Convolutional Neural Networks (CNNs) revolutionised face recognition. Modern systems use deep neural networks trained on millions of face images to learn discriminative face representations (embeddings) that are highly robust to variations in pose, lighting, expression, age, and occlusion. Notable systems include:

- **DeepFace (Facebook, 2014):** A nine-layer deep neural network that achieved near-human-level accuracy (97.35%) on the Labelled Faces in the Wild (LFW) benchmark.
- **FaceNet (Google, 2015):** Introduced the concept of learning a 128-dimensional Euclidean embedding of faces using a deep CNN optimised with a triplet loss function. FaceNet achieved a then-record accuracy of 99.63% on LFW.
- **dlib's ResNet Face Recognition Model:** This is the model used in our project through the `face_recognition` Python library. It uses a 29-layer ResNet (Residual Neural Network) trained using a triplet loss function on approximately 3 million face images. The model generates a 128-dimensional encoding for each face, and recognition is performed by computing the Euclidean distance between encodings. It achieves 99.38% accuracy on the LFW benchmark, making it one of the most accurate open-source face recognition models available.

**Related Work in Attendance Systems:**

Several research papers and projects have explored the application of face recognition for attendance management:

- Nandhini et al. (2022) developed an automated attendance system using OpenCV and the LBPH algorithm. While functional, the system's accuracy was limited (approximately 85%) due to the reliance on a traditional feature-extraction method and single-photo registration.
- Balaji et al. (2023) proposed a deep learning–based attendance system using CNN for feature extraction. Their system achieved 95% accuracy but required significant computational resources (GPU) for real-time processing, limiting its deployability in resource-constrained environments.
- The system presented in this project builds upon these prior works by combining the high accuracy of dlib's deep learning model with a multi-sample registration workflow (50 photos with diversity-based encoding selection), resulting in robust performance even on CPU-only hardware.

## 2.2 Existing System

The existing system for attendance management and monitoring in most educational institutions in India, particularly in the North-Eastern region, relies on the following manual processes:

1. **Manual Roll Call:** The faculty member reads out student names from a register at the beginning or end of each class. Students respond verbally, and the teacher marks their presence. This process is repeated for every lecture session throughout the day.
2. **Paper-Based Registers:** Attendance is recorded in physical registers that are maintained per class per subject. These registers are stored in the department office and are the official record of student attendance.
3. **Manual Compilation:** At the end of each month or semester, the attendance data from individual registers must be manually compiled and calculated to determine each student's attendance percentage. This is a labour-intensive process prone to arithmetic errors.
4. **CCTV Surveillance (Passive):** Many institutions have installed CCTV cameras in corridors and common areas. However, these systems only record video footage. There is no automated analysis or real-time alerting. The footage is typically reviewed only after a security incident has been reported.
5. **No Integration:** There is no integration between the attendance system and the surveillance system. They operate as completely independent, siloed processes.

**Disadvantages of the Existing System:**

- High time consumption for daily attendance.
- Vulnerability to proxy attendance.
- Human errors in data recording and compilation.
- Difficulty in generating consolidated reports and analytics.
- No real-time monitoring or alerting capability.
- Paper registers are susceptible to physical damage, loss, and tampering.
- Scalability is limited by human resources.

## 2.3 Proposed System

The proposed **AI Face Recognition and Monitoring System** is an integrated, intelligent platform that automates both attendance management and security monitoring using face recognition technology. The proposed system operates as follows:

1. **Face Registration:** An administrator registers new users (students or employees) by capturing 50 photographs of each person directly through the web browser's built-in camera. The photographs are saved to the server and the user is marked as "untrained."
2. **Model Training:** When the administrator clicks the "Train Model" button, the system processes all stored photographs for each untrained user, detects faces in each image, generates 128-dimensional face encodings using dlib's deep learning model, selects the top 5 most diverse encodings through a maximal diversity selection algorithm, and saves the trained encodings to a serialised file.
3. **Real-Time Recognition:** Once trained, the system can recognise registered individuals in real time from a live camera feed. The video stream is processed frame by frame: faces are detected, their encodings are computed, and these encodings are compared against the stored encodings. Recognised individuals are displayed with green bounding boxes and their names with confidence percentages. Unrecognised faces are displayed with red bounding boxes labelled "Unknown."
4. **Automated Attendance:** When a registered person is detected for the first time on a given day, their attendance is automatically marked in the database with the current date and time. The system determines whether the person is "Present" or "Late" based on a configurable time threshold (default: 9:30 AM). Duplicate entries for the same person on the same day are automatically prevented.
5. **Unknown Face Alerting:** When an unknown face is detected, the system captures a cropped snapshot of the face and saves it to the disk. A log entry is created in the database with a timestamp. A sophisticated multi-layer throttling mechanism ensures that the system does not generate an excessive number of alerts for the same unknown person lingering in the camera's field of view.
6. **Web-Based Administration:** All operations are managed through an intuitive, responsive web interface built with Flask and Bootstrap 5. The interface includes a dashboard with real-time statistics, a face registration page with camera capture and file upload tabs, an attendance records page with date-based filtering and CSV export, and an alerts page with interactive features.

**Advantages of the Proposed System:**

- Attendance is marked automatically and instantaneously, saving 5–10 minutes per lecture.
- Proxy attendance is completely eliminated, as the system requires the physical presence of the individual's face.
- All records are stored digitally in a database, enabling easy retrieval, analysis, and reporting.
- Real-time monitoring with intelligent alerting enhances security.
- CSV export enables integration with existing administrative workflows.
- The system is cost-effective, requiring only a standard computer with a webcam.
- The multi-photo training approach ensures high recognition accuracy across varying conditions.

## 2.4 Comparison: Existing vs. Proposed System

**Table 2.1: Comparison of Existing and Proposed Systems**

| Feature               | Existing System                | Proposed System            |
| --------------------- | ------------------------------ | -------------------------- |
| Attendance Method     | Manual roll call               | Automatic face recognition |
| Time Required         | 5–10 minutes per session      | Instantaneous (< 1 second) |
| Proxy Attendance      | Easily possible                | Not possible (biometric)   |
| Data Storage          | Paper-based registers          | Digital database (SQLite)  |
| Report Generation     | Manual compilation             | Automated with CSV export  |
| Monitoring            | Passive CCTV recording         | Real-time face recognition |
| Unknown Person Alert  | None                           | Automatic with throttling  |
| Accuracy              | Subject to human error         | 99.38% (dlib benchmark)    |
| Scalability           | Limited by human resources     | Scales with hardware       |
| Cost                  | Ongoing (stationery, manpower) | One-time (hardware, setup) |
| Data Durability       | Susceptible to damage/loss     | Secure digital storage     |
| Integration Potential | None                           | CSV export, API-ready      |

## 2.5 Feasibility Study

A feasibility study was conducted to evaluate the viability of the proposed system from three critical perspectives:

### 2.5.1 Technical Feasibility

The proposed system is technically feasible because:

- **Mature Libraries Available:** The core libraries required — OpenCV (version 4.8+), face_recognition (version 1.3+), dlib (version 19.24+), and Flask (version 2.3+) — are all mature, well-documented, open-source libraries with active community support. They are available for installation through Python's pip package manager.
- **Standard Hardware Sufficiency:** The system runs on a standard laptop or desktop computer with a webcam. No specialised hardware such as a GPU, depth sensor, or infrared camera is required. The HOG (Histogram of Oriented Gradients) face detection model used for real-time recognition is optimised for CPU execution and can process video frames at acceptable frame rates on modern commodity hardware.
- **Web-Based Architecture:** The Flask web framework provides a lightweight, easy-to-deploy web server that serves the user interface. The MJPEG streaming protocol used for the live video feed is supported by all modern web browsers without requiring any plugins or extensions.
- **Proven Algorithmic Accuracy:** The dlib face recognition model used in the system has been independently benchmarked at 99.38% accuracy on the LFW dataset, demonstrating its technical capability for the intended use case.

### 2.5.2 Economic Feasibility

The project is economically feasible because:

- **Zero Software Licensing Costs:** All software components — Python, OpenCV, face_recognition, dlib, Flask, SQLite, and the operating system (Linux) — are open-source and free of charge. There are no recurring licensing fees.
- **Minimal Hardware Investment:** The system can be deployed on any standard computer (₹25,000 – ₹35,000) with a built-in or USB webcam (₹500 – ₹2,000 if external). Most educational institutions already have such hardware available in their computer laboratories.
- **Reduced Operational Costs:** By automating attendance and monitoring, the system reduces the ongoing costs associated with manual processes, including the time cost of faculty performing roll calls, the cost of attendance registers and stationery, and the administrative effort required for data compilation.
- **Low Maintenance:** The system uses a file-based SQLite database that requires no separate database server, no dedicated database administrator, and no ongoing maintenance costs.

### 2.5.3 Operational Feasibility

The system is operationally feasible because:

- **User-Friendly Interface:** The web-based interface is designed with modern UI/UX principles using Bootstrap 5, featuring clear navigation, intuitive controls, real-time feedback, and responsive layouts. Administrators with basic computer literacy can operate the system after minimal training.
- **Simple Registration Process:** Adding a new user involves entering basic details (name, email, roll number, department, semester), clicking "Open Camera" and "Start Capture" to take 50 photos automatically, and then clicking "Train Model" to activate recognition. The entire process takes less than 2 minutes per user.
- **Minimal Disruption:** The system operates passively. Students do not need to perform any action — they simply walk into the room, and their attendance is marked automatically by the camera. This ensures minimal disruption to existing workflows and routines.
- **Institutional Acceptance:** Face recognition for attendance is increasingly being adopted by educational institutions across India. The technology is well-understood, socially acceptable, and does not raise significant ethical concerns when deployed for legitimate institutional purposes with appropriate privacy policies.

---

<div style="page-break-after: always;"></div>

# CHAPTER 3: SYSTEM REQUIREMENTS

## 3.1 Hardware Requirements

The hardware requirements for developing and deploying the AI Face Recognition and Monitoring System are minimal, as the system is designed to run on standard commodity hardware.

**Table 3.1: Hardware Requirements**

| Component         | Minimum Requirement                   | Recommended                       |
| ----------------- | ------------------------------------- | --------------------------------- |
| Processor (CPU)   | Intel Core i3 (6th Gen) or equivalent | Intel Core i5 (8th Gen) or better |
| RAM               | 4 GB                                  | 8 GB or more                      |
| Hard Disk Storage | 10 GB free space                      | 50 GB or more (for photos)        |
| Webcam            | 720p (HD) resolution                  | 1080p (Full HD) resolution        |
| Display           | 1366 × 768 resolution                | 1920 × 1080 resolution           |
| Network           | LAN/Wi-Fi (for web access)            | Ethernet (for stable connection)  |

**Notes:**

- The **processor** should support the execution of multi-threaded Python applications. Modern Intel i3 or AMD Ryzen 3 processors are sufficient for the HOG-based face detection model used in real-time recognition. If the CNN model is used for higher accuracy, a dedicated NVIDIA GPU with CUDA support is recommended.
- **RAM** of 4 GB is sufficient for basic operation with a small number of registered users (up to 50). For larger deployments with hundreds of registered users and multiple concurrent browser sessions, 8 GB or more is recommended.
- **Storage** requirements depend on the number of registered users. Each user's registration generates approximately 50 JPEG images (each approximately 50–100 KB), totalling 2.5–5 MB per user. The face_encodings.pkl file grows linearly with the number of encodings stored. The SQLite database remains lightweight (under 1 MB for typical usage).
- The **webcam** quality directly impacts recognition accuracy. A higher resolution webcam captures more facial detail, enabling better face detection — particularly for individuals standing at a distance from the camera.

## 3.2 Software Requirements

The software requirements span the development environment, runtime dependencies, and supporting tools.

**Table 3.2: Software Requirements**

| Category             | Software                          | Version                 |
| -------------------- | --------------------------------- | ----------------------- |
| Operating System     | Linux (Ubuntu 22.04 LTS or later) | 22.04+                  |
|                      | Windows 10/11 (also supported)    | 10+                     |
| Programming Language | Python                            | 3.10+                   |
| Web Framework        | Flask                             | 2.3+                    |
| Computer Vision      | OpenCV (opencv-python)            | 4.8+                    |
| Face Recognition     | face_recognition                  | 1.3+                    |
| Machine Learning     | dlib                              | 19.24+                  |
| Numerical Computing  | NumPy                             | 1.24+                   |
| Data Processing      | Pandas                            | 2.0+                    |
| Build Tool           | CMake                             | 3.27+                   |
| Database             | SQLite                            | 3.x (built into Python) |
| Web Browser          | Google Chrome / Mozilla Firefox   | Latest                  |
| Code Editor / IDE    | Visual Studio Code / PyCharm      | Any                     |

**Notes:**

- **dlib** is a critical dependency that requires CMake, a C++ compiler (g++ on Linux, MSVC on Windows), and BLAS/LAPACK libraries for linear algebra operations. On Linux systems, these can be installed with: `sudo apt install cmake g++ libopenblas-dev liblapack-dev libx11-dev`. The compilation of dlib from source typically takes 5–15 minutes on first installation.
- **SQLite** is included in Python's standard library (`sqlite3` module), so no separate database server installation is required.
- The web browser must be accessed at `http://localhost:5000` (not via the network IP address) when using the camera features, as modern browsers block camera access (`getUserMedia` API) on non-HTTPS connections for security reasons, with an exception made only for `localhost`.

## 3.3 Technology Stack

The complete technology stack used in this project is summarised below:

**Table 3.3: Technology Stack Summary**

| Layer                      | Technology             | Purpose                                             |
| -------------------------- | ---------------------- | --------------------------------------------------- |
| **Backend**          | Python 3.10+           | Core programming language                           |
|                            | Flask 2.3+             | Web application framework                           |
|                            | Jinja2                 | Server-side HTML templating engine                  |
|                            | SQLite 3               | Embeddable relational database                      |
|                            | Werkzeug               | WSGI utility library (file uploads, security)       |
| **Face Recognition** | OpenCV 4.8+            | Video capture, image processing, frame manipulation |
|                            | face_recognition 1.3+  | Face detection and encoding generation              |
|                            | dlib 19.24+            | Deep learning face recognition model (ResNet-29)    |
|                            | NumPy 1.24+            | Numerical array operations for face vectors         |
|                            | Pickle                 | Serialisation of face encoding data                 |
| **Frontend**         | HTML5                  | Page structure and semantic markup                  |
|                            | CSS3 (Custom)          | Styling with CSS custom properties and gradients    |
|                            | Bootstrap 5.3.2        | Responsive UI framework with grid system            |
|                            | Bootstrap Icons 1.11.1 | Scalable vector icon library                        |
|                            | JavaScript (ES6+)      | Client-side interactivity and camera operations     |
|                            | Google Fonts (Inter)   | Modern typography for UI                            |
| **Browser APIs**     | getUserMedia           | Browser-native camera access for registration       |
|                            | Canvas API             | Frame capture from video stream                     |
|                            | Fetch API              | Asynchronous server communication (AJAX)            |
| **Streaming**        | MJPEG                  | Motion JPEG streaming for live video feed           |

---

<div style="page-break-after: always;"></div>

# CHAPTER 4: SYSTEM DESIGN

## 4.1 System Architecture

The AI Face Recognition and Monitoring System follows a **client-server architecture** with a monolithic backend design. The system's architecture can be described at three levels:

**Presentation Layer (Frontend):** The user interface is rendered in the web browser using server-side Jinja2 templates combined with Bootstrap 5 for responsive styling. Client-side JavaScript handles interactive features such as camera capture, AJAX requests, and dynamic UI updates. The live video feed is delivered as an MJPEG (Motion JPEG) stream directly from the server.

**Application Layer (Backend):** The Flask web application serves as the central controller, handling all HTTP requests, managing user sessions, coordinating between the face recognition engine and the database, and rendering HTML pages. The backend is organised into four core modules:

- `app.py` — Route definitions and request handling.
- `face_rec_engine.py` — Face detection, encoding, recognition, and video streaming.
- `attendance_manager.py` — Attendance operations and report generation.
- `config.py` — Centralised configuration management.

**Data Layer (Storage):** Data persistence is handled through:

- **SQLite Database** (`database.db`) — Stores structured data including user registrations, attendance records, unknown face logs, and admin credentials.
- **File System** — Stores binary data including registered face photographs, unknown face snapshots, face encoding pickle files, and exported CSV reports.

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB BROWSER (CLIENT)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │  Login   │ │Dashboard │ │ Register │ │  Live Feed   │   │
│  │  Page    │ │          │ │  Face    │ │  (MJPEG)     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
│  ┌──────────────┐ ┌──────────────┐                          │
│  │  Attendance  │ │    Alerts    │                          │
│  │   Records    │ │   Page      │                          │
│  └──────────────┘ └──────────────┘                          │
│            │ getUserMedia │ Fetch API │ MJPEG Stream │       │
└────────────┼──────────────┼───────────┼──────────────┼──────┘
             │              │           │              │
             ▼              ▼           ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                 FLASK APPLICATION SERVER                      │
│  ┌──────────────────┐  ┌─────────────────────────────┐      │
│  │     app.py       │  │   face_rec_engine.py        │      │
│  │  (Routes &       │──│  (Detection, Recognition,   │      │
│  │   Controllers)   │  │   Encoding, Streaming)      │      │
│  └──────────────────┘  └─────────────────────────────┘      │
│  ┌──────────────────┐  ┌─────────────────────────────┐      │
│  │attendance_manager│  │       config.py              │      │
│  │  (Reports, Stats)│  │  (Settings & Parameters)    │      │
│  └──────────────────┘  └─────────────────────────────┘      │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌──────────────────┐  ┌──────────────────────────────┐     │
│  │  SQLite Database │  │    File System               │     │
│  │  ┌────────────┐  │  │  ┌────────────────────┐      │     │
│  │  │   users    │  │  │  │ registered_faces/  │      │     │
│  │  ├────────────┤  │  │  ├────────────────────┤      │     │
│  │  │ attendance │  │  │  │ unknown_faces/     │      │     │
│  │  ├────────────┤  │  │  ├────────────────────┤      │     │
│  │  │unknown_face│  │  │  │ face_encodings.pkl │      │     │
│  │  ├────────────┤  │  │  ├────────────────────┤      │     │
│  │  │   admin    │  │  │  │ reports/ (CSV)     │      │     │
│  │  └────────────┘  │  │  └────────────────────┘      │     │
│  └──────────────────┘  └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

**Figure 4.1: System Architecture Diagram**

## 4.2 Data Flow Diagrams (DFD)

### 4.2.1 Context-Level DFD (Level 0)

The context-level DFD shows the system as a single process with its external entities and the data flows between them.

```
                    ┌─────────────┐
                    │             │
    Login Credentials │           │ Dashboard Stats, Attendance Data,
    Registration Data ──▶         │ Alerts, CSV Reports, Flash Messages
    Training Command  │   AI Face │
    Camera Feed       │  Rec &    │──▶ Admin
    Filter/Export     │ Monitoring│
    Commands          │  System   │
                      │           │
                      └─────┬─────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
                ┌───────┐     ┌────────┐
                │Webcam │     │Database│
                │Camera │     │ (Data  │
                │       │     │ Store) │
                └───────┘     └────────┘
```

**Figure 4.2: Context-Level DFD (Level 0)**

**External Entities:**

- **Admin:** The system administrator who interacts with the system through the web browser.
- **Webcam Camera:** The hardware device providing the live video feed for face capture and recognition.

### 4.2.2 Level 1 Data Flow Diagram

The Level 1 DFD decomposes the central process into its major sub-processes.

```
                         ┌──────────┐
                         │  Admin   │
                         └────┬─────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐     ┌─────────────┐     ┌─────────────┐
    │   1.0     │     │    2.0      │     │    3.0      │
    │  Admin    │     │   Face      │     │  Live Feed  │
    │  Login    │     │Registration │     │ & Recognition│
    │           │     │ & Training  │     │             │
    └─────┬─────┘     └──────┬──────┘     └──────┬──────┘
          │                  │                   │
          ▼                  ▼                   ▼
    ┌───────────┐     ┌─────────────┐     ┌─────────────┐
    │  Admin    │     │   Users     │     │ Attendance  │
    │  Table    │     │   Table     │     │   Table     │
    │  (D1)     │     │   (D2)     │     │   (D3)      │
    └───────────┘     └─────────────┘     └──────┬──────┘
                                                 │
                              ┌───────────────────┤
                              ▼                   ▼
                        ┌───────────┐     ┌─────────────┐
                        │   4.0     │     │    5.0      │
                        │ Attendance│     │  Unknown    │
                        │ Management│     │ Face Alerts │
                        └───────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │ Unknown     │
                                          │ Faces Table │
                                          │   (D4)      │
                                          └─────────────┘
```

**Figure 4.3: Level 1 Data Flow Diagram**

**Process Descriptions:**

| Process | Name                         | Description                                                                                                                                                            |
| ------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | Admin Login                  | Authenticates the administrator using credentials stored in the admin table. Creates a session upon successful authentication.                                         |
| 2.0     | Face Registration & Training | Handles the capture or upload of face photographs, stores them in the file system, creates user records in the database, and generates face encodings during training. |
| 3.0     | Live Feed & Recognition      | Captures video frames from the webcam, detects and recognises faces, draws overlays on the frame, and returns the annotated MJPEG stream.                              |
| 4.0     | Attendance Management        | Records attendance when a recognised person is detected, filters and retrieves records, and exports data to CSV format.                                                |
| 5.0     | Unknown Face Alerts          | Captures and stores snapshots of unrecognised faces, logs them in the database, and provides management features (review, delete, bulk operations).                    |

## 4.3 Entity-Relationship (ER) Diagram

The ER diagram below illustrates the logical data model of the system, showing the entities, their attributes, and the relationships between them.

```
┌──────────────────────────────────────────┐
│                 ADMIN                     │
├──────────────────────────────────────────┤
│ PK  admin_id    INTEGER (Auto)           │
│     username    TEXT (UNIQUE, NOT NULL)   │
│     password    TEXT (SHA-256 Hash)       │
│     email       TEXT                     │
│     created_date TIMESTAMP               │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│                 USERS                     │
├──────────────────────────────────────────┤
│ PK  user_id        INTEGER (Auto)        │
│     name           TEXT (NOT NULL)        │
│     email          TEXT                   │
│     roll_no        TEXT                   │
│     department     TEXT                   │
│     semester       TEXT                   │
│     photo_path     TEXT                   │
│     photo_dir      TEXT                   │
│     trained        INTEGER (0 or 1)       │
│     registered_date TIMESTAMP             │
└────────────┬─────────────────────────────┘
             │  1
             │
             │ has many
             │
             │  N
┌────────────┴─────────────────────────────┐
│              ATTENDANCE                   │
├──────────────────────────────────────────┤
│ PK  record_id     INTEGER (Auto)         │
│ FK  user_id       INTEGER                │
│     name          TEXT (NOT NULL)         │
│     date          TEXT (NOT NULL)         │
│     check_in_time TEXT (NOT NULL)         │
│     status        TEXT (Present/Late)     │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│            UNKNOWN_FACES                  │
├──────────────────────────────────────────┤
│ PK  log_id         INTEGER (Auto)        │
│     image_path     TEXT                   │
│     timestamp      TIMESTAMP              │
│     alert_status   TEXT (Unreviewed/      │
│                         Reviewed)         │
└──────────────────────────────────────────┘
```

**Figure 4.4: Entity-Relationship (ER) Diagram**

**Relationships:**

- **USERS → ATTENDANCE:** One-to-Many. A single user can have multiple attendance records (one per day). The `user_id` in the ATTENDANCE table is a foreign key referencing the `user_id` in the USERS table.
- **ADMIN** operates independently and is not linked to other entities via foreign keys. It stores the administrator credentials for authentication purposes.
- **UNKNOWN_FACES** operates independently. Unknown face records are not linked to any user because, by definition, the person is unrecognised.

## 4.4 Use Case Diagram

The Use Case diagram identifies the primary actors and the use cases (functions) available to them.

```
                        ┌──────────────────────────────────────────┐
                        │          AI Face Recognition &            │
                        │           Monitoring System              │
                        │                                          │
┌──────┐               │  ┌─────────────────────────────┐        │
│      │  ─────────────── │      Login / Logout          │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │   View Dashboard             │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │   Register Face (Camera)     │        │
│ Admin│               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │   Register Face (Upload)     │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │   Train Recognition Model    │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │  Start/Stop Live Feed        │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │  View/Filter Attendance      │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │  Export Attendance (CSV)      │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │  View/Manage Alerts          │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
│      │  ─────────────── ┌─────────────────────────────┐        │
│      │               │  │  Edit/Delete Users           │        │
│      │               │  └─────────────────────────────┘        │
│      │               │                                          │
└──────┘               └──────────────────────────────────────────┘
```

**Figure 4.5: Use Case Diagram**

## 4.5 Module Description

The system is composed of five interconnected modules:

### Module 1: Authentication Module (`app.py` — routes: `/`, `/login`, `/logout`)

The Authentication Module is responsible for managing administrator access to the system. It provides a secure login page where the administrator enters their username and password. The entered password is hashed using the SHA-256 algorithm and compared against the stored hash in the admin table of the database. Upon successful authentication, a server-side session is created using Flask's session management (backed by signed cookies). The `login_required` decorator function is applied to all protected routes to ensure that only authenticated administrators can access the system's features. The logout functionality clears the session, stops any active camera feed, and redirects the user to the login page.

### Module 2: Face Registration and Training Module (`app.py`, `face_rec_engine.py`)

This module handles the complete lifecycle of user registration, from data entry and photo capture to model training. It supports two registration methods:

**Camera-Based Registration:** The administrator opens the user's browser webcam using the `getUserMedia` API, which provides a live video preview. Upon clicking "Start Capture," the system automatically captures 50 frames at 350ms intervals (approximately 17.5 seconds total), displaying a real-time progress bar and frame counter. The captured frames are sent to the server as Base64-encoded JPEG images via a JSON POST request to the `/register_camera` endpoint. The server decodes each frame, saves it as a JPEG file in the user's designated directory, and creates a database record with the user's information and a `trained=0` flag.

**File Upload Registration:** The administrator fills in the user's details and uploads a single frontal-face photograph. The server validates the file type, saves it to the registered_faces directory, creates a database record, and immediately generates a face encoding.

**Model Training:** The "Train Model" panel displays all untrained users (those with `trained=0`). When the administrator clicks the "Train Model" button, the system iterates through each untrained user, reads all their stored photographs, detects faces using the HOG model, and generates 128-dimensional face encodings with configurable `num_jitters` (re-sampling count). From all generated encodings, the top N (configurable, default 5) most diverse encodings are selected using a greedy farthest-point selection algorithm, which maximises the minimum distance between selected encodings. This diversity-based selection ensures that the stored encodings capture the full range of a person's facial variations. The selected encodings are appended to the in-memory lists and serialised to a pickle file.

### Module 3: Real-Time Recognition and Monitoring Module (`face_rec_engine.py`, `app.py`)

This module handles the core face recognition functionality and live video streaming. The `FaceRecognitionEngine` class maintains the live camera feed using OpenCV's `VideoCapture`, processes each frame through the recognition pipeline, draws annotated overlays (bounding boxes, names, confidence scores, and timestamps), and yields the processed frames as an MJPEG byte stream through Flask's streaming response mechanism.

The recognition pipeline operates as follows for each frame:

1. The frame is resized by a configurable factor (default 0.5) for faster processing.
2. Histogram equalisation is applied to the luminance channel (YUV colour space) to normalise lighting conditions.
3. Face locations are detected using the HOG-based detector.
4. Face encodings are generated for each detected face.
5. Each encoding is compared against all known encodings using Euclidean distance.
6. If the minimum distance is below the configured tolerance (default 0.4), the person is identified.
7. Coordinates are scaled back to the original frame size, and results are returned.

For performance optimisation, recognition is performed only on every third frame. The results are cached and reused for drawing overlays on the intermediate frames, ensuring smooth visual output without recognition lag.

### Module 4: Attendance Management Module (`attendance_manager.py`, `database/db_handler.py`)

This module handles all attendance-related operations. When the recognition module identifies a registered person, it calls `mark_attendance()`, which checks whether an attendance record already exists for that person on the current date. If no record exists, a new entry is created with the current timestamp and a status of either "Present" or "Late" (based on the configurable threshold, defaulting to 9:30 AM). Duplicate prevention is a crucial feature that ensures only one attendance record per person per day, regardless of how many times they appear in the camera feed.

The module also provides functionality to retrieve attendance records with optional date-based filtering, compute attendance statistics (total, present, late counts), and export records to CSV files. The CSV export includes columns for Record ID, User ID, Name, Date, Check-in Time, and Status.

### Module 5: Unknown Face Alert Module (`face_rec_engine.py`, `database/db_handler.py`, `app.py`)

This module manages the detection, logging, and review of unrecognised faces. When the recognition module detects a face that does not match any stored encoding (or matches with a distance exceeding the tolerance threshold), the face is classified as "Unknown." To prevent the system from generating an overwhelming number of alerts (e.g., if an unknown person stands in front of the camera for an extended period), a sophisticated four-layer throttling mechanism is employed:

1. **Minimum Face Size Filter:** Faces smaller than 60×60 pixels (configurable) are ignored, as they are likely too distant for reliable identification or alerting.
2. **Spatial-Bucket Cooldown:** The frame is divided into spatial "buckets" of 80×80 pixels (configurable). Each bucket tracks the last time an unknown face was saved from that region. A new save from the same bucket is suppressed for 300 seconds (5 minutes, configurable).
3. **Encoding-Based Deduplication:** A ring buffer of the last 20 unknown face encodings is maintained. Before saving a new unknown face, its encoding is compared against all recent encodings using face distance. If the minimum distance is below the recognition tolerance, the face is considered a duplicate of a previously saved unknown and is not saved again.
4. **Global Rate Cap:** A maximum of 3 unknown face saves per minute (configurable) is enforced as a hard limit, regardless of other filters.

Only after passing all four filters is an unknown face image cropped (with a 20-pixel margin), saved to the `unknown_faces/` directory, and logged in the database. The alerts page provides an interactive grid view of all alerts with filter buttons (All/Unreviewed/Reviewed), hover zoom effects, a full-screen lightbox modal with image download and delete capabilities, select-all functionality, and bulk delete operations.

## 4.6 Database Design

The system uses a **SQLite** relational database for data persistence. SQLite was chosen because it is embedded within Python's standard library, requires no separate server process, stores the entire database in a single file, and provides ACID-compliant transactions — making it ideal for a standalone application.

The database consists of four tables:

**Table 4.1: Users Table Schema**

| Column Name     | Data Type | Constraints                | Description                              |
| --------------- | --------- | -------------------------- | ---------------------------------------- |
| user_id         | INTEGER   | PRIMARY KEY, AUTOINCREMENT | Unique identifier for each user          |
| name            | TEXT      | NOT NULL                   | Full name of the user                    |
| email           | TEXT      | —                         | Email address (optional)                 |
| roll_no         | TEXT      | —                         | University roll number                   |
| department      | TEXT      | —                         | Department (e.g., BCA, CSE)              |
| semester        | TEXT      | —                         | Current semester (e.g., 6)               |
| photo_path      | TEXT      | —                         | Relative path to the preview photo       |
| photo_dir       | TEXT      | —                         | Relative path to the photo directory     |
| trained         | INTEGER   | DEFAULT 0                  | Training status (0=Untrained, 1=Trained) |
| registered_date | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP  | Registration timestamp                   |

**Table 4.2: Attendance Table Schema**

| Column Name   | Data Type | Constraints                   | Description                        |
| ------------- | --------- | ----------------------------- | ---------------------------------- |
| record_id     | INTEGER   | PRIMARY KEY, AUTOINCREMENT    | Unique attendance record ID        |
| user_id       | INTEGER   | FOREIGN KEY → users(user_id) | ID of the recognised user          |
| name          | TEXT      | NOT NULL                      | Name of the user                   |
| date          | TEXT      | NOT NULL                      | Date of attendance (YYYY-MM-DD)    |
| check_in_time | TEXT      | NOT NULL                      | Time of first detection (HH:MM:SS) |
| status        | TEXT      | DEFAULT 'Present'             | Present or Late                    |

**Table 4.3: Unknown Faces Table Schema**

| Column Name  | Data Type | Constraints                | Description                           |
| ------------ | --------- | -------------------------- | ------------------------------------- |
| log_id       | INTEGER   | PRIMARY KEY, AUTOINCREMENT | Unique alert record ID                |
| image_path   | TEXT      | —                         | Relative path to the saved face image |
| timestamp    | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP  | Date and time of detection            |
| alert_status | TEXT      | DEFAULT 'Unreviewed'       | Unreviewed or Reviewed                |

**Table 4.4: Admin Table Schema**

| Column Name  | Data Type | Constraints                | Description                |
| ------------ | --------- | -------------------------- | -------------------------- |
| admin_id     | INTEGER   | PRIMARY KEY, AUTOINCREMENT | Unique admin ID            |
| username     | TEXT      | UNIQUE, NOT NULL           | Admin username             |
| password     | TEXT      | NOT NULL                   | SHA-256 hashed password    |
| email        | TEXT      | —                         | Admin email (optional)     |
| created_date | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP  | Account creation timestamp |

---

<div style="page-break-after: always;"></div>

# CHAPTER 5: IMPLEMENTATION

## 5.1 Implementation Overview

The implementation of the AI Face Recognition and Monitoring System was carried out using the **Python** programming language (version 3.10+) as the primary development language, with the **Flask** micro-framework for the web application layer. The project follows a modular design where each major functionality is encapsulated in a dedicated Python module, promoting code reusability, maintainability, and separation of concerns.

The development process followed an iterative approach:

1. **Phase 1:** Database design and initialization, admin authentication.
2. **Phase 2:** Face registration (file upload) and basic face encoding.
3. **Phase 3:** Live camera feed integration and real-time recognition.
4. **Phase 4:** Automated attendance logging with duplicate prevention.
5. **Phase 5:** Camera-based multi-photo registration and two-step training workflow.
6. **Phase 6:** Unknown face detection with multi-layer throttling.
7. **Phase 7:** Frontend polish (dashboard, alerts lightbox, responsive design) and testing.

## 5.2 Project Directory Structure

The project is organised into the following directory structure:

```
Face_Rec_Mon/
│
├── app.py                        # Main Flask application — all routes
│                                 #   (login, register, attendance, alerts,
│                                 #   video feed, train model, user management)
│
├── face_rec_engine.py            # Core face recognition engine — loads/saves
│                                 #   encodings, detects faces, draws overlays,
│                                 #   streams MJPEG, captures frames
│
├── attendance_manager.py         # High-level attendance operations — marks
│                                 #   attendance, generates stats, exports CSV
│
├── config.py                     # Centralised configuration — file paths,
│                                 #   recognition tolerance, model selection,
│                                 #   throttling parameters, server settings
│
├── requirements.txt              # Python package dependencies
│
├── database/
│   ├── __init__.py               # Package initializer
│   ├── db_handler.py             # SQLite database handler — all CRUD
│   │                             #   operations for users, attendance,
│   │                             #   unknown faces, and admin
│   └── database.db               # SQLite database file (auto-created)
│
├── encodings/
│   └── face_encodings.pkl        # Serialised face encodings (auto-created
│                                 #   after model training)
│
├── static/
│   ├── css/
│   │   └── style.css             # Custom application stylesheet with
│   │                             #   CSS variables, gradients, animations
│   ├── registered_faces/         # Captured photos organised per user
│   │   └── <User_Name>/          #   (50 JPEG images per user)
│   └── unknown_faces/            # Cropped snapshots of unrecognised faces
│
├── templates/
│   ├── base.html                 # Shared layout — sidebar navigation,
│   │                             #   flash messages, Bootstrap/Icons CDN
│   ├── login.html                # Admin login page with animated card
│   ├── dashboard.html            # Overview — stat cards, recent attendance,
│   │                             #   quick action buttons
│   ├── register.html             # Face registration (camera + upload tabs),
│   │                             #   registered users table, edit modal,
│   │                             #   train model panel
│   ├── live_feed.html            # Live video feed with start/stop controls,
│   │                             #   fullscreen mode, status indicator
│   ├── attendance.html           # Attendance records with date filter,
│   │                             #   summary stats, CSV export button
│   └── alerts.html               # Unknown face alerts — card grid, filter
│                                 #   buttons, lightbox modal, bulk delete
│
├── reports/                      # Exported CSV attendance reports
│
├── .gitignore                    # Git ignore rules for runtime data
│
└── README.md                     # Project overview and usage instructions
```

**Figure 5.1: Project Directory Structure**

## 5.3 Core Backend Modules

### 5.3.1 Configuration Module (`config.py`)

The configuration module centralises all system parameters in a single file. This design ensures that adjustments (such as changing the recognition tolerance or the face detection model) can be made in one place without modifying any other module. The key configuration parameters are:

**Table 5.1: Configuration Parameters and Their Descriptions**

| Parameter                    | Default Value                | Description                                          |
| ---------------------------- | ---------------------------- | ---------------------------------------------------- |
| `SECRET_KEY`               | (random string)              | Flask session signing key                            |
| `DEBUG`                    | True                         | Flask debug mode                                     |
| `HOST`                     | '0.0.0.0'                    | Server bind address                                  |
| `PORT`                     | 5000                         | Server port                                          |
| `DATABASE_PATH`            | database/database.db         | SQLite database file path                            |
| `ENCODINGS_FILE`           | encodings/face_encodings.pkl | Serialised encodings file path                       |
| `RECOGNITION_TOLERANCE`    | 0.4                          | Maximum face distance for a match (lower = stricter) |
| `MODEL`                    | 'hog'                        | Face detection model for live feed ('hog' or 'cnn')  |
| `TRAIN_MODEL`              | 'hog'                        | Face detection model for training                    |
| `FRAME_RESIZE_FACTOR`      | 0.5                          | Frame downscale ratio for processing speed           |
| `NUM_JITTERS`              | 3                            | Number of re-samples per face for encoding averaging |
| `NUM_ENCODINGS_PER_PERSON` | 5                            | Number of diverse encodings stored per user          |
| `UNKNOWN_COOLDOWN_SECONDS` | 300                          | Spatial cooldown duration (seconds)                  |
| `UNKNOWN_BUCKET_SIZE`      | 80                           | Spatial bucket size in pixels                        |
| `UNKNOWN_MIN_FACE_SIZE`    | 60                           | Minimum face dimension for unknown saves             |
| `LATE_THRESHOLD_HOUR`      | 9                            | Hour component of the late threshold                 |
| `LATE_THRESHOLD_MINUTE`    | 30                           | Minute component of the late threshold               |

### 5.3.2 Database Handler Module (`database/db_handler.py`)

The database handler module encapsulates all SQLite database operations. It follows the **Data Access Object (DAO)** pattern, where each database operation is implemented as a method of the `DatabaseHandler` class. The module handles:

- **Database Initialization:** The `init_db()` method creates all four tables (users, attendance, unknown_faces, admin) if they do not already exist. It also performs automatic schema migration by detecting missing columns and adding them with `ALTER TABLE` statements, ensuring backward compatibility with older database versions.
- **Admin Operations:** The `verify_admin()` method hashes the provided password with SHA-256 and queries the admin table for a matching username-password pair.
- **User CRUD Operations:** Methods for adding users (`add_user()`), retrieving all users (`get_all_users()`), finding users by name or ID, deleting users, updating user information, and managing the training status.
- **Attendance Operations:** The `mark_attendance()` method implements the duplicate prevention logic by checking for existing records before inserting. The `get_attendance()` and `get_recent_attendance()` methods support optional date filtering and limit clauses.
- **Unknown Faces Operations:** Methods for logging unknown face detections, retrieving alerts, updating alert status (mark as reviewed), and single/bulk deletion of alert records.

All database connections use SQLite's `Row` factory for dictionary-style access to query results.

### 5.3.3 Face Recognition Engine (`face_rec_engine.py`)

The `FaceRecognitionEngine` class is the central intelligence of the system. It manages the face encodings data (stored as Python lists of NumPy arrays), the camera hardware interface, and the recognition pipeline. Key methods include:

- **`load_encodings()` / `save_encodings()`:** Deserialise/serialise the known face encodings, names, and user IDs from/to a pickle file.
- **`register_face()`:** Loads an image file, detects faces, generates an encoding, and appends it to the known encodings list.
- **`recognize_faces()`:** Performs the complete recognition pipeline on a video frame (resize → histogram equalisation → face detection → encoding computation → distance comparison → result compilation).
- **`draw_results()`:** Renders bounding boxes (green for recognised, red for unknown), name labels, and confidence percentages onto the video frame using OpenCV drawing functions.
- **`generate_frames()`:** A Python generator function that continuously reads frames from the webcam, runs recognition on every third frame (caching results for smooth overlay rendering), handles attendance marking and unknown face logging, adds timestamp overlays, JPEG-encodes each processed frame, and yields the result as an MJPEG byte stream.
- **`generate_preview_frames()`:** A simplified camera stream for the registration page that draws a guide oval (showing where to position the face) without running recognition.
- **`capture_single_frame()`:** Opens the camera, captures a single frame (with warm-up frames for exposure adjustment), and returns it as a Base64-encoded data URL.

## 5.4 Frontend Design and Templates

The frontend is built using the **Jinja2** templating engine (integrated with Flask) and **Bootstrap 5.3.2** for responsive styling. A custom CSS stylesheet (`style.css`) extends Bootstrap with a bespoke design system featuring:

- **CSS Custom Properties (Variables):** The stylesheet defines a comprehensive set of design tokens including sidebar dimensions, colour palette, gradient definitions, shadow scales, border radius values, and transition timings.
- **Gradient Stat Cards:** The dashboard uses three gradient-coloured stat cards (blue/purple for users, green for attendance, pink/red for alerts) with decorative pseudo-element overlays and hover animations.
- **Animated Login Page:** The login page features a full-screen gradient background, decorative circle pseudo-elements, a glassmorphic login card with backdrop blur, and a `slideUp` CSS keyframe animation.
- **Responsive Sidebar:** The navigation sidebar adapts to smaller screens by collapsing to an icon-only mode at widths below 768px.
- **Custom Scrollbar:** The scrollbar is styled with a narrow track and rounded thumb for a modern appearance.

The template hierarchy uses Jinja2 template inheritance:

- `base.html` — Master layout with sidebar navigation, flash message rendering, Bootstrap and icon CDN links, and content blocks.
- Individual page templates (`dashboard.html`, `register.html`, `attendance.html`, `alerts.html`, `live_feed.html`, `login.html`) extend `base.html` and fill in their respective content blocks.

## 5.5 Face Recognition Pipeline

The face recognition pipeline is the technical core of the system. The following flowchart illustrates the step-by-step process:

```
┌──────────────────────┐
│   Camera captures    │
│     video frame      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Resize frame by     │
│  FRAME_RESIZE_FACTOR │
│     (default 0.5)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Convert to YUV      │
│  Apply histogram     │
│  equalisation on Y   │
│  Convert back to BGR │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Convert BGR → RGB   │
│  Detect face         │
│  locations (HOG)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Generate 128-dim    │
│  face encoding for   │
│  each detected face  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Compare encoding    │
│  against all known   │
│  encodings (Euclidean│
│  distance)           │
└──────────┬───────────┘
           │
      ┌────┴────┐
      │         │
      ▼         ▼
┌──────────┐ ┌──────────┐
│ Distance │ │ Distance │
│ ≤ 0.4    │ │ > 0.4    │
│ MATCH    │ │ UNKNOWN  │
└────┬─────┘ └────┬─────┘
     │             │
     ▼             ▼
┌──────────┐ ┌──────────┐
│  Mark    │ │ Apply    │
│attendance│ │throttle  │
│(once/day)│ │filters & │
│          │ │save alert│
└──────────┘ └──────────┘
```

**Figure 5.2: Face Recognition Pipeline Flowchart**

## 5.6 Key Algorithms

### 5.6.1 Maximal Diversity Encoding Selection Algorithm

When training the model, the system captures up to 50 face encodings per user but stores only the top 5 most diverse encodings. The selection uses a **greedy farthest-point selection** algorithm:

```
Algorithm: Maximal Diversity Selection
Input: List of all encodings E (up to 50), desired count N (default 5)
Output: Selected subset S of N encodings

1. S ← {E[0]}  (start with the first encoding)
2. R ← {1, 2, ..., |E|-1}  (remaining indices)
3. WHILE |S| < N AND R is not empty:
   a. FOR each index i in R:
      - Compute min_dist = MIN(||E[i] - s|| for all s in S)
   b. best_idx ← argmax(min_dist over all i in R)
   c. S ← S ∪ {E[best_idx]}
   d. R ← R \ {best_idx}
4. RETURN S
```

This algorithm ensures that the selected encodings are as dissimilar as possible from each other, capturing the maximum variance in a person's facial appearance.

### 5.6.2 Unknown Face Throttling Algorithm

The four-layer throttling mechanism for unknown faces operates as follows:

```
Algorithm: Unknown Face Throttle
Input: Detected unknown face with location (top, right, bottom, left) and encoding
Output: Decision to save or suppress the alert

1. face_w ← right - left; face_h ← bottom - top
2. IF face_w < MIN_SIZE OR face_h < MIN_SIZE:
   SUPPRESS (face too small)

3. bucket_x ← center_x // BUCKET_SIZE
   bucket_y ← center_y // BUCKET_SIZE
   key ← "bucket_x_bucket_y"
4. IF cooldown[key] exists AND (now - cooldown[key]) < COOLDOWN_SECONDS:
   SUPPRESS (spatial cooldown active)

5. IF recent_encodings is not empty:
   distances ← face_distance(recent_encodings, encoding)
   IF min(distances) ≤ TOLERANCE:
      cooldown[key] ← now
      SUPPRESS (duplicate encoding)

6. timestamps ← filter(timestamps, within last 60 seconds)
   IF count(timestamps) ≥ MAX_SAVES_PER_MINUTE:
   SUPPRESS (rate limit exceeded)

7. SAVE face image and log alert
   cooldown[key] ← now
   timestamps.append(now)
   recent_encodings.append(encoding)
```

**Figure 5.4: Unknown Face Throttling Mechanism**

---

<div style="page-break-after: always;"></div>

# CHAPTER 6: TESTING

## 6.1 Testing Strategy

Testing is a critical phase in the software development lifecycle that ensures the system functions correctly, reliably, and as per the specified requirements. The testing strategy for the AI Face Recognition and Monitoring System was designed to cover all functional modules, edge cases, and user interaction scenarios.

The testing was conducted manually following a structured approach:

1. Each module was tested independently (unit testing).
2. Modules were tested in combination to verify inter-module communication (integration testing).
3. The complete system was tested end-to-end to verify overall functionality (system testing).
4. The system was demonstrated to peers and the project guide for usability feedback (user acceptance testing).

## 6.2 Types of Testing Performed

### 6.2.1 Unit Testing

Each module was tested individually to verify that its functions produce the expected output for given inputs. For example:

- The `DatabaseHandler.add_user()` method was tested to confirm that it correctly inserts a new user record and returns a valid user ID.
- The `DatabaseHandler.mark_attendance()` method was tested to verify that it prevents duplicate entries for the same user on the same day.
- The `FaceRecognitionEngine.recognize_faces()` method was tested with known face images to confirm correct identification.

### 6.2.2 Integration Testing

Integration testing verified the interaction between modules:

- The registration workflow was tested end-to-end: camera capture → frame saving → database insertion → model training → live recognition.
- The attendance pipeline was tested: live recognition → `mark_attendance()` call → database insertion → dashboard statistics update.
- The alert pipeline was tested: unknown face detection → throttle filter evaluation → image saving → database logging → alerts page display.

### 6.2.3 System Testing

System testing evaluated the complete application as a whole:

- All web pages were accessed and tested for correct rendering, navigation, and responsiveness.
- The login/logout flow was tested with valid and invalid credentials.
- The CSV export was verified for correct data content and file format.
- Error handling was tested by intentionally triggering error conditions (e.g., submitting a form without required fields, uploading a file with no face visible).

### 6.2.4 User Acceptance Testing (UAT)

The system was demonstrated to fellow students and the project guide to gather feedback on:

- Ease of use and intuitiveness of the interface.
- Accuracy of face recognition under typical indoor lighting conditions.
- Speed of the registration and training process.
- Clarity and usefulness of the dashboard and attendance reports.

## 6.3 Test Cases and Results

**Table 6.1: Test Cases and Results**

| Test Case ID | Test Description                             | Input / Action                                     | Expected Result                                  | Actual Result                                           | Status  |
| ------------ | -------------------------------------------- | -------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------- | ------- |
| TC-01        | Admin login with valid credentials           | Username: admin, Password: admin123                | Redirect to dashboard with welcome message       | Redirected to dashboard, flash message displayed        | ✅ Pass |
| TC-02        | Admin login with invalid credentials         | Username: admin, Password: wrongpass               | Error message, stay on login page                | Flash message "Invalid username or password" displayed  | ✅ Pass |
| TC-03        | Login with empty fields                      | Username: (empty), Password: (empty)               | Error message prompting input                    | Flash message "Please enter both username and password" | ✅ Pass |
| TC-04        | Access protected page without login          | Navigate to /dashboard directly                    | Redirect to login page                           | Redirected to login with warning message                | ✅ Pass |
| TC-05        | Register user via camera capture (50 photos) | Enter name "Test User", open camera, start capture | 50 frames captured, user saved as untrained      | 50 frames saved, user appeared in untrained list        | ✅ Pass |
| TC-06        | Register user without entering name          | Click "Open Camera" without name                   | Warning message displayed                        | "Please enter a name before opening the camera" shown   | ✅ Pass |
| TC-07        | Register user via file upload                | Upload a clear frontal face photo                  | User registered with face encoding generated     | User saved and encoding created successfully            | ✅ Pass |
| TC-08        | Upload file with no face visible             | Upload a landscape photo with no face              | Warning about no face detected                   | "No face detected in the photo" warning displayed       | ✅ Pass |
| TC-09        | Upload invalid file type                     | Upload a .txt file                                 | Error message about file type                    | "Invalid file type. Use JPG or PNG" error displayed     | ✅ Pass |
| TC-10        | Train model with untrained users             | Click "Train Model" with pending users             | Model trained, users marked as trained           | Encodings generated, users moved to trained status      | ✅ Pass |
| TC-11        | Train model with no untrained users          | Click "Train Model" with no pending users          | Appropriate message displayed                    | "No untrained users found" message shown                | ✅ Pass |
| TC-12        | Live feed recognises registered person       | Start feed, show trained person's face             | Green box with name and confidence               | Person correctly identified with 85%+ confidence        | ✅ Pass |
| TC-13        | Live feed detects unknown person             | Start feed, show unregistered person's face        | Red box with "Unknown" label                     | "Unknown" label displayed with red bounding box         | ✅ Pass |
| TC-14        | Automatic attendance marking                 | Show registered face to camera (first time today)  | Attendance record created with correct timestamp | Record created in database with "Present" status        | ✅ Pass |
| TC-15        | Duplicate attendance prevention              | Show same person to camera again (same day)        | No duplicate entry created                       | Second detection did not create new record              | ✅ Pass |
| TC-16        | Late attendance marking                      | Show face after 9:30 AM                            | Attendance marked as "Late"                      | Status correctly set to "Late"                          | ✅ Pass |
| TC-17        | View attendance with date filter             | Select a specific date and click search            | Only records for that date displayed             | Filtered records displayed correctly                    | ✅ Pass |
| TC-18        | Export attendance to CSV                     | Click "Export CSV" button                          | CSV file downloaded with correct data            | CSV downloaded with all attendance columns              | ✅ Pass |
| TC-19        | View unknown face alerts                     | Navigate to Alerts page                            | All alerts displayed in card grid                | Alerts displayed with thumbnails and timestamps         | ✅ Pass |
| TC-20        | Mark alert as reviewed                       | Click "Mark Reviewed" on an unreviewed alert       | Status changed to "Reviewed"                     | Alert status updated, badge changed                     | ✅ Pass |
| TC-21        | Delete single alert                          | Click "Delete" on an alert card                    | Alert and image file deleted                     | Both database record and image file removed             | ✅ Pass |
| TC-22        | Bulk delete alerts                           | Select multiple alerts, click bulk delete          | All selected alerts and images deleted           | All selected items removed successfully                 | ✅ Pass |
| TC-23        | Edit user information                        | Click edit, modify roll number, save               | User info updated in database                    | Updated information visible in table                    | ✅ Pass |
| TC-24        | Delete registered user                       | Click delete on a registered user                  | User, photos, and encodings removed              | All traces of user completely removed                   | ✅ Pass |
| TC-25        | Logout functionality                         | Click "Logout" in sidebar                          | Session cleared, redirect to login               | Redirected to login, camera stopped                     | ✅ Pass |
| TC-26        | Responsive layout on mobile                  | Access on a mobile-width browser                   | Sidebar collapses, content adapts                | Sidebar collapsed to icons, content remained usable     | ✅ Pass |
| TC-27        | Alert lightbox modal                         | Click on an alert image                            | Full-screen lightbox opens                       | Lightbox opened with download and delete options        | ✅ Pass |
| TC-28        | Camera access on localhost                   | Access via http://localhost:5000                   | Camera access granted by browser                 | getUserMedia succeeded, camera feed displayed           | ✅ Pass |

---

<div style="page-break-after: always;"></div>

# CHAPTER 7: SCREENSHOTS AND OUTPUT

> **Note to the Student:** Replace the descriptions below with actual screenshots from your running system. To take screenshots:
>
> 1. Run the application with `python app.py`
> 2. Open `http://localhost:5000` in your browser
> 3. Navigate to each page and capture screenshots using the Print Screen key or a screenshot tool
> 4. Name the image files as indicated (e.g., `fig_7_1_login.png`) and insert them into the printed document

## 7.1 Login Page

**Figure 7.1: Login Page**

The login page features a gradient background (blue to purple), a centered glassmorphic login card with an animated slide-up entrance effect, the application branding (FaceRecMon icon and title), and fields for username and password input. The default credentials (admin / admin123) are displayed at the bottom of the card for first-time setup convenience.

*[INSERT SCREENSHOT: Login page with gradient background and login form]*

## 7.2 Dashboard

**Figure 7.2: Dashboard Page**

The dashboard provides an at-a-glance overview of the system's status. It displays three gradient-coloured stat cards showing the total number of registered users, today's attendance count, and the number of unreviewed unknown face alerts. Below the stat cards, the dashboard is divided into two sections: a "Quick Actions" panel on the left with shortcut buttons for common tasks (Register New Face, View Attendance, View Alerts, Export CSV), and a "Recent Attendance" table on the right showing the latest 10 attendance entries with their names, dates, times, and status badges.

*[INSERT SCREENSHOT: Dashboard with stat cards and recent attendance table]*

## 7.3 Face Registration — Camera Capture

**Figure 7.3: Face Registration — Camera Capture**

The registration page features a tabbed interface with "Live Camera" and "Upload Photo" tabs. The camera tab shows input fields for the user's Full Name (required), Email, Roll No, Department, and Semester. When the camera is opened, a live video preview is displayed with a mirrored view. During the 50-photo capture process, a real-time progress bar shows the capture progress, a frame counter badge overlays the video feed, and flash effects animate with each captured frame. A thumbnail strip below the video shows every 5th captured frame for visual verification.

*[INSERT SCREENSHOT: Registration page during camera capture with progress bar]*

## 7.4 Face Registration — File Upload

**Figure 7.4: Face Registration — File Upload**

The upload tab provides a traditional form-based registration interface. The administrator fills in the user's details and selects a clear, front-facing photograph (JPG or PNG format). An image preview is displayed after file selection. Clicking "Register Face" uploads the photo, saves the user record, and immediately generates a face encoding.

*[INSERT SCREENSHOT: Registration page showing file upload tab with photo preview]*

## 7.5 Train Model Panel

**Figure 7.5: Train Model Panel**

The Train Model panel is located below the registration section. It displays an informational banner explaining the two-step workflow, a table of all untrained users (showing their photo thumbnail, name, email, and registration date), and a "Train Model" button. During training, an animated progress bar and status text are displayed. Upon completion, a success message lists all trained users with the number of samples processed and encodings stored.

*[INSERT SCREENSHOT: Train Model panel showing untrained users and the train button]*

## 7.6 Live Feed with Recognition

**Figure 7.6: Live Feed with Real-Time Face Recognition**

The Live Feed page features a large video display area with start/stop controls, a status indicator (Offline/Live), and a real-time clock. When the feed is active, the video stream shows detected faces with coloured bounding boxes: green boxes with the person's name and confidence percentage for recognised individuals, and red boxes with the "Unknown" label for unrecognised faces. A timestamp overlay in the top-left corner of the video shows the current date and time. Fullscreen mode is available for projector or monitor display.

*[INSERT SCREENSHOT: Live feed showing faces with green and red bounding boxes]*

## 7.7 Attendance Records

**Figure 7.7: Attendance Records Page**

The attendance page displays a date filter input on the left, three summary stat cards (Total Records, Present, Late) on the right, and a comprehensive attendance table showing Record ID, Name, Date, Check-in Time, and Status (with colour-coded badges — green for "Present," yellow for "Late"). An "Export CSV" button in the top-right corner allows downloading the filtered data.

*[INSERT SCREENSHOT: Attendance records page with summary stats and data table]*

## 7.8 Unknown Face Alerts

**Figure 7.8: Unknown Face Alerts Page**

The alerts page displays a card grid of all unknown face detections. Each card shows a cropped face thumbnail, a status ribbon (red "New" for unreviewed, grey "Reviewed" for reviewed alerts), a timestamp, and action buttons (Mark Reviewed, Delete). The top of the page features filter buttons (All / Unreviewed / Reviewed), a select-all checkbox, and a bulk delete button that appears when items are selected. Hovering over a card thumbnail reveals a zoom effect, and clicking it opens a full-screen lightbox modal with download and delete options.

*[INSERT SCREENSHOT: Alerts page showing card grid with filter buttons]*

**Figure 7.9: Alert Lightbox Modal**

*[INSERT SCREENSHOT: Full-screen lightbox modal showing enlarged unknown face image]*

## 7.9 CSV Export Sample

**Figure 7.10: CSV Export Output**

The exported CSV file contains columns for Record ID, User ID, Name, Date, Check-in Time, and Status. The file is automatically named based on the filter criteria (e.g., `attendance_2026-04-13.csv` for a date-specific export or `attendance_all_20260413.csv` for a full export).

*[INSERT SCREENSHOT or paste of CSV content in a text editor]*

---

<div style="page-break-after: always;"></div>

# CHAPTER 8: CONCLUSION AND FUTURE SCOPE

## 8.1 Conclusion

The **AI Face Recognition and Monitoring System** has been successfully designed, developed, and tested as a comprehensive solution for automated attendance management and intelligent security monitoring. The project demonstrates the practical application of modern computer vision and deep learning technologies in solving real-world problems faced by educational institutions.

The following objectives were achieved through this project:

1. **Real-Time Face Recognition:** The system successfully detects and recognises registered individuals from a live camera feed in real time, with recognition accuracy exceeding 95% under typical indoor lighting conditions. The use of dlib's deep learning model (ResNet-29) for generating 128-dimensional face encodings provides state-of-the-art recognition performance.
2. **Automated Attendance:** The system automatically logs attendance when a registered person is detected by the camera, completely eliminating the need for manual roll calls. The duplicate prevention mechanism ensures data integrity by allowing only one attendance record per person per day.
3. **Robust Registration Workflow:** The two-step registration and training workflow — capturing 50 photographs per user and selecting the 5 most diverse encodings through a maximal diversity algorithm — significantly improves recognition accuracy compared to single-photo systems. This approach captures the natural variance in a person's facial appearance across different expressions, angles, and lighting conditions.
4. **Security Monitoring:** The unknown face detection and alerting module provides an effective security layer by automatically identifying and logging unrecognised individuals. The four-layer throttling mechanism ensures that alerts are meaningful and manageable, preventing the system from being overwhelmed by repeated detections of the same unknown person.
5. **User-Friendly Web Interface:** The Flask-based web dashboard provides an intuitive, modern, and responsive interface for system administration. The use of Bootstrap 5, Inter font, gradient stat cards, animated login page, and interactive features (lightbox modals, inline editing, bulk operations) delivers a professional and pleasant user experience.
6. **Cost-Effective Deployment:** The system runs entirely on open-source libraries and standard hardware, making it accessible and affordable for educational institutions with limited budgets. No commercial software licenses or specialised hardware are required.

## 8.2 Limitations

Despite its successful implementation, the system has the following limitations that should be acknowledged:

1. **Single Camera Support:** The current implementation is designed to work with a single camera at a time. Multi-camera deployments would require modifications to the camera management and streaming architecture.
2. **No Anti-Spoofing:** The system does not implement facial liveness detection. It is theoretically possible to deceive the system using a high-resolution printed photograph or a video playback of a registered person's face. Anti-spoofing measures such as blink detection, texture analysis, or depth sensing would address this vulnerability.
3. **Lighting Sensitivity:** While histogram equalisation improves robustness, extreme lighting conditions (very dark rooms, strong backlighting, or uneven illumination) can still affect recognition accuracy. The system performs best under consistent indoor lighting.
4. **Occlusion Sensitivity:** Face masks, sunglasses, or other facial occlusions can prevent face detection and recognition. The system is not designed to recognise partially occluded faces.
5. **Scalability Constraints:** While the system handles hundreds of registered users effectively, the linear comparison approach (computing Euclidean distance against all stored encodings) may become a bottleneck with thousands of registrations. More efficient indexing structures (e.g., KD-trees or approximate nearest neighbour search) would be needed for large-scale deployments.
6. **No Cloud Integration:** The system operates as a standalone, locally deployed application. It does not synchronise data to a cloud server or support remote access beyond the local network.

## 8.3 Future Scope

The AI Face Recognition and Monitoring System provides a solid foundation for numerous future enhancements:

1. **Multi-Camera Support:** Extending the system to handle multiple camera feeds simultaneously, allowing coverage of multiple classrooms, entrances, or areas from a single dashboard.
2. **Anti-Spoofing / Liveness Detection:** Implementing facial liveness detection techniques (blink detection, challenge-response, depth analysis using 3D cameras) to prevent spoofing attacks.
3. **Mask-Aware Recognition:** Integrating face recognition models that are specifically trained to recognise individuals wearing face masks, using the upper-face region (forehead, eyes, eyebrows).
4. **Mobile Application:** Developing a companion mobile application (Android/iOS) that allows administrators to monitor the system, receive push notifications for unknown face alerts, and view attendance reports remotely.
5. **Cloud Integration:** Moving the database and backend to a cloud platform (such as AWS, Google Cloud, or Microsoft Azure) to enable centralised data storage, remote access, automatic backups, and multi-site deployment.
6. **Integration with College ERP:** Developing APIs to integrate attendance data with existing college ERP systems, Learning Management Systems (LMS), or student information systems for seamless data flow.
7. **Attendance Analytics and Reporting:** Building advanced analytics dashboards with graphical reports (attendance trends, class-wise comparisons, chronic absenteeism identification) and automated email notifications to parents/guardians when a student's attendance falls below a threshold.
8. **GPU-Accelerated Processing:** Utilising NVIDIA CUDA-enabled GPUs to switch from the HOG face detection model to the CNN model, significantly improving detection accuracy and enabling real-time processing of higher-resolution video feeds.
9. **Emotion Detection:** Incorporating facial emotion recognition to monitor student engagement during lectures, providing instructors with insights into the emotional state of their class.
10. **Edge Deployment with Raspberry Pi:** Optimising the system for deployment on single-board computers like the Raspberry Pi, enabling cost-effective, distributed deployment across multiple locations.

---

<div style="page-break-after: always;"></div>

# CHAPTER 9: REFERENCES AND BIBLIOGRAPHY

## Books

1. Bradski, G. and Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media, Inc. ISBN: 978-0-596-51613-0.
2. Grinberg, M. (2018). *Flask Web Development: Developing Web Applications with Python*. 2nd Edition. O'Reilly Media, Inc. ISBN: 978-1-491-99173-2.
3. Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press. ISBN: 978-0-262-03561-3.
4. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3rd Edition. O'Reilly Media, Inc. ISBN: 978-1-098-12597-4.
5. Lutz, M. (2013). *Learning Python*. 5th Edition. O'Reilly Media, Inc. ISBN: 978-1-449-35573-9.

## Research Papers

6. Turk, M.A. and Pentland, A.P. (1991). "Eigenfaces for Recognition." *Journal of Cognitive Neuroscience*, 3(1), pp. 71-86.
7. Schroff, F., Kalenichenko, D., and Philbin, J. (2015). "FaceNet: A Unified Embedding for Face Recognition and Clustering." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 815-823.
8. Taigman, Y., Yang, M., Ranzato, M., and Wolf, L. (2014). "DeepFace: Closing the Gap to Human-Level Performance in Face Verification." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1701-1708.
9. King, D.E. (2009). "Dlib-ml: A Machine Learning Toolkit." *Journal of Machine Learning Research*, 10, pp. 1755-1758.
10. Dalal, N. and Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection." *IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, vol. 1, pp. 886-893.

## Online References

11. OpenCV Official Documentation. Available at: https://docs.opencv.org/ [Accessed: March 2026].
12. Face Recognition Library Documentation. Available at: https://face-recognition.readthedocs.io/ [Accessed: March 2026].
13. dlib C++ Library. Available at: http://dlib.net/ [Accessed: March 2026].
14. Flask Official Documentation. Available at: https://flask.palletsprojects.com/ [Accessed: March 2026].
15. SQLite Official Documentation. Available at: https://www.sqlite.org/docs.html [Accessed: March 2026].
16. Bootstrap 5 Documentation. Available at: https://getbootstrap.com/docs/5.3/ [Accessed: March 2026].
17. MDN Web Docs — getUserMedia API. Available at: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia [Accessed: March 2026].
18. Python Official Documentation — sqlite3 Module. Available at: https://docs.python.org/3/library/sqlite3.html [Accessed: March 2026].
19. NumPy Official Documentation. Available at: https://numpy.org/doc/ [Accessed: March 2026].
20. Labelled Faces in the Wild Benchmark. Available at: http://vis-www.cs.umass.edu/lfw/ [Accessed: March 2026].

---

<div style="page-break-after: always;"></div>

# CHAPTER 10: APPENDIX

## 10.1 Source Code — Key Modules

> **Note:** Only the key portions of the source code are included here. The complete source code is available in the project submission folder (soft copy).

### 10.1.1 Main Application (`app.py`) — Key Routes

```python
"""
AI Face Recognition and Monitoring System
Main Flask Application
"""
import os
import base64
import json
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for,
                   Response, session, flash, send_file, jsonify)
from werkzeug.utils import secure_filename
from config import (SECRET_KEY, DATABASE_PATH, ENCODINGS_FILE,
                    REGISTERED_FACES_DIR, UNKNOWN_FACES_DIR,
                    REPORTS_DIR, RECOGNITION_TOLERANCE, MODEL,
                    TRAIN_MODEL, NUM_JITTERS, NUM_ENCODINGS_PER_PERSON,
                    DEBUG, HOST, PORT)
from database.db_handler import DatabaseHandler
from face_rec_engine import FaceRecognitionEngine, FACE_REC_AVAILABLE
from attendance_manager import AttendanceManager

# ── Initialize Flask App ──────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ── Initialize Components ────────────────────────────────────────
db = DatabaseHandler(DATABASE_PATH)
engine = FaceRecognitionEngine(ENCODINGS_FILE, RECOGNITION_TOLERANCE, MODEL)
att_mgr = AttendanceManager(db, REPORTS_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Decorator to require admin login."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def index():
    """Login page (redirects to dashboard if already logged in)."""
    if 'admin' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    """Handle admin login."""
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    if not username or not password:
        flash('Please enter both username and password.', 'error')
        return redirect(url_for('index'))
    if db.verify_admin(username, password):
        session['admin'] = username
        flash(f'Welcome back, {username}!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid username or password.', 'error')
        return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with statistics and recent activity."""
    stats = att_mgr.get_today_stats()
    recent = db.get_recent_attendance(limit=10)
    return render_template('dashboard.html',
                           stats=stats, recent=recent,
                           face_rec_available=FACE_REC_AVAILABLE)
```

### 10.1.2 Face Recognition Engine (`face_rec_engine.py`) — Core Methods

```python
class FaceRecognitionEngine:
    """Core engine for face detection and recognition."""

    def __init__(self, encodings_path, tolerance=0.6, model='hog'):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.model = model
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.is_running = False
        self.camera = None
        self.load_encodings()

    def recognize_faces(self, frame):
        """Detect and recognize faces in a video frame."""
        if not FACE_REC_AVAILABLE:
            return []

        # Resize for faster processing
        fx = self.resize_factor
        small_frame = cv2.resize(frame, (0, 0), fx=fx, fy=fx)

        # Histogram equalization for better lighting handling
        yuv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        preprocessed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        rgb_frame = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_frame, model=self.model
        )
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations
        )

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            confidence = 0.0
            user_id = None

            if len(self.known_encodings) > 0:
                face_distances = face_recognition.face_distance(
                    self.known_encodings, encoding
                )
                best_match_idx = np.argmin(face_distances)
                best_distance = face_distances[best_match_idx]

                if best_distance <= self.tolerance:
                    name = self.known_names[best_match_idx]
                    confidence = round(1.0 - best_distance, 2)
                    if best_match_idx < len(self.known_ids):
                        user_id = self.known_ids[best_match_idx]

            # Scale back coordinates
            sb = self.scale_back
            top, right, bottom, left = location
            top *= sb; right *= sb; bottom *= sb; left *= sb

            results.append({
                'name': name, 'user_id': user_id,
                'location': (top, right, bottom, left),
                'confidence': confidence, 'encoding': encoding
            })

        return results
```

### 10.1.3 Database Handler (`database/db_handler.py`) — Attendance Methods

```python
def mark_attendance(self, name, user_id=None):
    """Mark attendance for a recognized person (once per day)."""
    today = date.today().strftime('%Y-%m-%d')
    now = datetime.now().strftime('%H:%M:%S')
    conn = self.get_connection()
    cursor = conn.cursor()

    # Prevent duplicate entries for the same day
    cursor.execute(
        'SELECT * FROM attendance WHERE name=? AND date=?',
        (name, today)
    )
    if cursor.fetchone() is not None:
        conn.close()
        return False  # Already marked

    # Determine status (Present / Late)
    hour = datetime.now().hour
    minute = datetime.now().minute
    from config import LATE_THRESHOLD_HOUR, LATE_THRESHOLD_MINUTE
    if hour > LATE_THRESHOLD_HOUR or (
        hour == LATE_THRESHOLD_HOUR and minute > LATE_THRESHOLD_MINUTE
    ):
        status = 'Late'
    else:
        status = 'Present'

    cursor.execute(
        '''INSERT INTO attendance
           (user_id, name, date, check_in_time, status)
           VALUES (?, ?, ?, ?, ?)''',
        (user_id, name, today, now, status)
    )
    conn.commit()
    conn.close()
    return True
```

## 10.2 Installation Guide

### Step 1: Install System Dependencies (Linux/Ubuntu)

```bash
sudo apt update
sudo apt install cmake g++ libopenblas-dev liblapack-dev libx11-dev
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/Nkingsap/Face_Rec_Mon.git
cd Face_Rec_Mon
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `dlib` library compiles from source and may take 5–15 minutes on first installation.

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Access the Application

Open a web browser and navigate to:

```
http://localhost:5000
```

### Default Login Credentials

| Username | Password |
| -------- | -------- |
| admin    | admin123 |

> **Important:** Always use `localhost` (not your IP address) to enable browser camera access over HTTP.

## 10.3 Glossary of Terms

| Term                                              | Definition                                                                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **AI (Artificial Intelligence)**            | A branch of computer science concerned with building systems capable of performing tasks that normally require human intelligence.    |
| **API (Application Programming Interface)** | A set of protocols and definitions for building and integrating application software.                                                 |
| **AJAX (Asynchronous JavaScript and XML)**  | A technique for creating asynchronous web requests without reloading the entire page.                                                 |
| **Base64**                                  | A binary-to-text encoding scheme that represents binary data in an ASCII string format.                                               |
| **Biometric**                               | A measurable physical or behavioural characteristic used for identity verification (e.g., face, fingerprint, iris).                   |
| **Bootstrap**                               | A free, open-source CSS framework for developing responsive, mobile-first websites.                                                   |
| **CNN (Convolutional Neural Network)**      | A class of deep learning neural network architecture commonly used in image and video analysis.                                       |
| **CSV (Comma-Separated Values)**            | A text file format that uses commas to separate values, commonly used for data exchange.                                              |
| **CUDA**                                    | A parallel computing platform and API created by NVIDIA for general-purpose GPU programming.                                          |
| **dlib**                                    | A C++ library with Python bindings containing machine learning algorithms and tools, including face detection and recognition models. |
| **DFD (Data Flow Diagram)**                 | A graphical representation of the flow of data through an information system.                                                         |
| **ER Diagram**                              | A graphical representation of entities, their attributes, and their relationships in a database.                                      |
| **Encoding (Face)**                         | A 128-dimensional numerical vector that represents the unique features of a person's face.                                            |
| **Flask**                                   | A lightweight WSGI web application framework in Python, designed for simplicity and flexibility.                                      |
| **getUserMedia**                            | A browser API that requests permission to access the user's camera and/or microphone.                                                 |
| **HOG (Histogram of Oriented Gradients)**   | A feature descriptor used for object detection, particularly effective for human face and body detection.                             |
| **Jinja2**                                  | A template engine for Python, used in Flask for rendering HTML pages with dynamic content.                                            |
| **LFW (Labelled Faces in the Wild)**        | A benchmark dataset of face photographs for evaluating face verification algorithms.                                                  |
| **MJPEG (Motion JPEG)**                     | A video compression format where each video frame is separately compressed as a JPEG image.                                           |
| **NumPy**                                   | A Python library for large, multi-dimensional arrays and matrices, with mathematical functions.                                       |
| **OpenCV**                                  | An open-source computer vision and machine learning software library with over 2,500 optimised algorithms.                            |
| **Pickle**                                  | A Python module for serialising (converting to byte stream) and deserialising Python objects.                                         |
| **ResNet (Residual Neural Network)**        | A deep learning architecture that uses "skip connections" to enable training of very deep networks.                                   |
| **SHA-256**                                 | A cryptographic hash function that generates a 256-bit (32-byte) hash value, used for password hashing.                               |
| **SQLite**                                  | A self-contained, serverless, zero-configuration SQL database engine embedded within the application.                                 |
| **Tolerance**                               | A threshold value (default 0.4) representing the maximum acceptable Euclidean distance between face encodings for a positive match.   |
| **WSGI (Web Server Gateway Interface)**     | A specification for a universal interface between web servers and web applications in Python.                                         |

---

<center>

**— END OF PROJECT REPORT —**

</center>
