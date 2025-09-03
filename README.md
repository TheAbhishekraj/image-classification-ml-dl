
***

# Image Classification ML/DL Project (Spoon vs Fork)

## Overview
A deep learning project to classify images as either **spoon** or **fork** using a Convolutional Neural Network (CNN) in TensorFlow/Keras. This project targets more than 95% accuracy and includes scripts for training and prediction.

## Folder Structure

```
data/
  spoon/     # Place spoon images here
  fork/      # Place fork images here
src/
  train_model.py      # Model training script
  predict.py          # Script for making predictions
utils/
  data_utils.py       # Helper for image processing/augmentation
notebooks/
  train_model.ipynb   # Jupyter Notebook for training and experimenting
models/               # Saved Keras models (.h5) go here
results/              # Training results, metrics, plots
requirements.txt      # Python dependencies list
README.md             # Project overview and instructions
LICENSE               # Usage license
```

## Getting Started

1. **Add images:**  
   - Put spoon images in `data/spoon/`
   - Put fork images in `data/fork/`
   - At least 30 images per class recommended.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python src/train_model.py
   ```
   - The model will be saved in the `models/` folder.

4. **Make predictions:**
   ```bash
   python src/predict.py --model models/spoon_fork_detector.h5 --image path/to/image.jpg
   ```
   or for a folder:
   ```bash
   python src/predict.py --model models/spoon_fork_detector.h5 --folder path/to/images/
   ```

## Notes

- Use sharp, clean images with variety in lighting, angle, and type.
- Commit your changes after every major update.
- If only a ZIP is uploaded, download and unzip locally to access all files.

## License

This project is licensed under the MIT License. See LICENSE for details.

***
