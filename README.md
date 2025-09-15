# Detecting Parkinson's Application

## Overview
The **Detecting Parkinson's Application** is a machine learning-based tool designed to assist in the early detection of Parkinson's disease by analyzing audio data. The application processes audio files, extracts relevant features, preprocesses data, trains ensemble models (stacking and voting), and provides a web interface for users to perform self-diagnosis. The project utilizes a dataset of synthesized vowel recordings from healthy controls and patients with Parkinson's disease and related disorders.

## Workflow
The application follows these steps:
1. **Audio Segmentation**: The `split.py` script splits audio files in the `audio_files` directory into 5-second segments and saves metadata to `segmented_info.csv`.
2. **Feature Extraction**: The `extract_features.py` script extracts audio features from the segmented files and saves them to `extracted_features.csv`.
3. **Data Preprocessing and Model Training**: The `parkinson_classification.ipynb` notebook in the `notebook` directory preprocesses the extracted features, visualizes the data, trains stacking and voting ensemble models, and saves the cleaned data to `cleaned_features.csv` along with the trained models.
4. **Web Interface**: The `app.py` script launches a web application allowing users to input audio data and receive a Parkinson's disease diagnosis.

## Features
- Splits audio files into 5-second segments for consistent analysis.
- Extracts audio features such as jitter, shimmer, harmonic-to-noise ratio (HNR), and subharmonics, relevant to Parkinson's disease detection.
- Preprocesses data to ensure quality and compatibility with machine learning models, including visualization of distributions and relationships.
- Trains ensemble models (stacking and voting) for robust predictions.
- Provides a user-friendly web interface for self-diagnosis.

## Dataset
The project uses the **Synthetic Vowels of Speakers with Parkinson’s Disease and Parkinsonism** dataset, available at [Figshare](https://figshare.com/articles/dataset/Synthetic_vowels_of_speakers_with_Parkinson_s_disease_and_Parkinsonism/7628819). This dataset, published by Jan Hlavnička et al. on October 29, 2019, contains synthesized replicas of sustained vowels (/A/ and /I/) from:
- Healthy controls (HC)
- Patients with Parkinson’s disease (PD)
- Patients with multiple system atrophy (MSA)
- Patients with progressive supranuclear palsy (PSP)

### Dataset Details
- **Purpose**: The dataset is designed for evaluating pitch detectors, modal fundamental frequency detectors, and subharmonic detectors.
- **File Naming**: Each recording is named with a code `Uvxy`, where:
  - `U`: Group (HC, PD, MSA, PSP)
  - `v`: Numeric identifier for the subject
  - `x`: Vowel type (a = /A/, i = /I/)
  - `y`: Repetition number
- **File Types** (e.g., for record `HC8a1`):
  - `HC8a1.wav`: Synthesized waveform with noise.
  - `HC8a1_clean.wav`: Synthesized waveform without noise.
  - `HC8a1_LF.wav`: Glottal pulse used for synthesis.
  - `HC8a1_impulses.csv`: Impulse locations and amplitudes for jitter and shimmer calculations.
  - `HC8a1_subharmonics.csv`: Subharmonic intervals with amplitude modulation (SHR) in percent (if applicable).
- **Metadata**: The `dataset.csv` file describes all recordings, including jitter, shimmer, HNR, and SHR values.
- **Usage**: Place the dataset audio files (e.g., `*.wav`) in the `audio_files` directory for processing by the application.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnhDuy2808/Detecting_Parkinson_Application.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Detecting_Parkinson_Application
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Ensure Python 3.x is installed. Install Jupyter separately to run the notebook:
   ```bash
   pip install jupyter
   ```

## Usage
1. **Prepare Audio Files**: Download the dataset from [Figshare](https://figshare.com/articles/dataset/Synthetic_vowels_of_speakers_with_Parkinson_s_disease_and_Parkinsonism/7628819) and place the audio files (e.g., `*.wav`) in the `audio_files` directory.
2. **Segment Audio**: Run the segmentation script:
   ```bash
   python src/split.py
   ```
   This generates `segmented_info.csv`.
3. **Extract Features**: Run the feature extraction script:
   ```bash
   python src/extract_features.py
   ```
   This generates `extracted_features.csv`.
4. **Preprocess Data and Train Models**: Open and run the `parkinson_classification.ipynb` notebook in the `notebook` directory to preprocess the data, visualize it, train the ensemble models, and generate `cleaned_features_no_outliers.csv`.
5. **Launch Web App**: Start the web application:
   ```bash
   python app.py
   ```
   Access the web interface (typically at `http://localhost:5000`) to perform diagnosis.

## Directory Structure
- `audio_files/`: Directory for input audio files (e.g., from the dataset).
- `notebook/`: Contains `parkinson_classification.ipynb` for data preprocessing, visualization, and model training.
- `data/segmented_info.csv`: Stores metadata of segmented audio files.
- `data/extracted_features.csv`: Stores extracted audio features.
- `data/cleaned_features.csv`: Stores preprocessed features.
- `src/split.py`: Script for audio segmentation.
- `src/extract_features.py`: Script for feature extraction.
- `app.py`: Script for launching the web interface.
- `requirements.txt`: Lists project dependencies.

## Dependencies
The project requires the following Python packages (as specified in `requirements.txt`):
- Flask==3.1.2
- joblib==1.5.0
- librosa==0.11.0
- nolds==0.6.2
- numpy==2.3.3
- pandas==2.3.2
- parselmouth==1.1.1
- pydub==0.25.1
- scikit-learn==1.7.2
- scipy==1.16.2
- soundfile==0.13.1
- tqdm==4.67.1
- xgboost==3.0.5

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## Acknowledgments
- The dataset used in this project is provided by Jan Hlavnička, Roman Čmejla, Jiří Klempíř, Evžen Růžička, and Jan Rusz. Citation: Hlavnička, J., Čmejla, R., Klempíř, J., Růžička, E., & Rusz, J. (2019). Synthetic vowels of speakers with Parkinson’s disease and Parkinsonism. Figshare. [https://figshare.com/articles/dataset/Synthetic_vowels_of_speakers_with_Parkinson_s_disease_and_Parkinsonism/7628819](https://figshare.com/articles/dataset/Synthetic_vowels_of_speakers_with_Parkinson_s_disease_and_Parkinsonism/7628819).