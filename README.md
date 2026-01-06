# Pre-trained Image Classifier for Dog Breed Identification

A Python application that uses pre-trained Convolutional Neural Network (CNN) models to classify pet images, with a focus on identifying dog breeds. This project evaluates and compares the performance of three different CNN architectures (ResNet, AlexNet, and VGG) to determine which provides the most accurate classification results.

## Project Overview

This application was developed to assist a citywide dog show registration system by automatically verifying whether submitted pet images are actually dogs. The classifier not only distinguishes dogs from non-dogs but also identifies specific dog breeds, enabling the detection of fraudulent registrations.

### Objectives

1. Correctly identify which pet images are of dogs (even if the breed is misclassified) and which pet images are not dogs
2. Correctly classify the breed of dog for images that are of dogs
3. Determine which CNN model architecture (ResNet, AlexNet, or VGG) best achieves objectives 1 and 2
4. Consider the time resources required and determine if an alternative solution provides acceptable results given runtime constraints

## Architecture

The application leverages transfer learning by using pre-trained CNN models from PyTorch's torchvision library. These models were trained on ImageNet, a dataset containing 1.2 million images across 1,000 categories, including 118 different dog breeds.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT PROCESSING                               │
│                           (check_images.py)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌─────────────┐             ┌─────────────┐             ┌────────────────┐
│  Command    │             │  Pet Image  │             │   Image        │
│  Line Args  │             │   Labels    │             │ Classification │
│  Parsing    │             │  Extraction │             │   (CNN)        │
└─────────────┘             └─────────────┘             └────────────────┘
       │                           │                           │
       │    get_input_args.py      │   get_pet_labels.py       │  classify_images.py
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Results Dictionary     │
                    │  (Pet Label, Classifier     │
                    │   Label, Match Status)      │
                    └─────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
         ┌─────────────────┐           ┌─────────────────┐
         │  Dog/Not-Dog    │           │   Statistics    │
         │  Classification │           │   Calculation   │
         └─────────────────┘           └─────────────────┘
                    │                             │
                    │ adjust_results4_isadog.py   │ calculates_results_stats.py
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Results Output        │
                    │     (print_results.py)      │
                    └─────────────────────────────┘
```

## Project Structure

```
├── check_images.py              # Main program orchestrating the classification pipeline
├── get_input_args.py            # Command line argument parsing
├── get_pet_labels.py            # Extract pet labels from image filenames
├── classify_images.py           # Image classification using CNN models
├── adjust_results4_isadog.py    # Determine if labels are dogs or not
├── calculates_results_stats.py  # Calculate classification statistics
├── print_results.py             # Format and display results
├── classifier.py                # CNN classifier function (ResNet, AlexNet, VGG)
├── dognames.txt                 # Reference list of all valid dog breed names
├── imagenet1000_clsid_to_human.txt  # ImageNet class ID to label mapping
├── run_models_batch.sh          # Batch script to run all three models
├── run_models_batch_uploaded.sh # Batch script for uploaded images
├── pet_images/                  # Directory containing 40 pet images for testing
├── uploaded_images/             # Directory for user-uploaded test images
└── *_pet-images.txt             # Output results for each model architecture
```

## CNN Model Architectures

The application supports three pre-trained CNN architectures:

| Model | Description | ImageNet Top-5 Accuracy |
|-------|-------------|------------------------|
| **VGG-16** | 16-layer deep network with small 3x3 convolution filters | 92.7% |
| **AlexNet** | 8-layer network, winner of ImageNet 2012 | 79.1% |
| **ResNet-18** | 18-layer network with residual connections | 89.1% |

## Results

Performance comparison across all three model architectures on the 40-image test dataset:

| Metric | ResNet | AlexNet | VGG |
|--------|--------|---------|-----|
| % Correct Dogs | 100% | 100% | 100% |
| % Correct Not-Dogs | 90% | 100% | 100% |
| % Correct Breed | 90% | 80% | 93.3% |

### Key Findings

**Best Overall Model: VGG**

VGG outperformed both other architectures when considering both primary objectives:
- Achieved 100% accuracy in identifying dogs vs. non-dogs
- Achieved the highest breed classification accuracy at 93.3%
- While AlexNet also achieved 100% on dog identification, its breed classification was lower at 80%

### Misclassification Analysis

Common misclassifications observed across models:
- **Great Pyrenees** misclassified as **Kuvasz** (visually similar breeds)
- **Beagle** misclassified as **Walker Hound** (closely related breeds)
- Some cat images misclassified as wild cats (e.g., lynx)

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- PIL (Pillow)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dog-breed-classifier.git
cd dog-breed-classifier
```

2. Install dependencies:
```bash
pip install torch torchvision pillow
```

## Usage

### Basic Usage

Run the classifier with default settings (VGG model):
```bash
python check_images.py
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dir` | Path to folder containing images | `pet_images/` |
| `--arch` | CNN model architecture (`resnet`, `alexnet`, `vgg`) | `vgg` |
| `--dogfile` | Text file containing valid dog names | `dognames.txt` |

### Examples

Classify images using ResNet:
```bash
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt
```

Classify custom images using AlexNet:
```bash
python check_images.py --dir uploaded_images/ --arch alexnet --dogfile dognames.txt
```

### Batch Processing

Run all three models and save results to text files:
```bash
sh run_models_batch.sh
```

This generates:
- `resnet_pet-images.txt`
- `alexnet_pet-images.txt`
- `vgg_pet-images.txt`

## Output Format

The program outputs a comprehensive summary including:

```
*** Results Summary for CNN Model Architecture VGG ***
N Images            :  40
N Dog Images        :  30
N Not-Dog Images    :  10

Percentage Stats:
pct_match           : 87.50
pct_correct_dogs    : 100.00
pct_correct_breed   : 93.33
pct_correct_notdogs : 100.00

** Total Elapsed Runtime: 0:0:47
```

## Data Dictionary

### Results Dictionary Structure

The results dictionary uses image filenames as keys with a list containing:

| Index | Description | Type |
|-------|-------------|------|
| 0 | Pet image label (from filename) | string |
| 1 | Classifier label (from CNN) | string |
| 2 | Label match (1=match, 0=no match) | int |
| 3 | Pet image is-a-dog (1=dog, 0=not dog) | int |
| 4 | Classifier is-a-dog (1=dog, 0=not dog) | int |

### Statistics Dictionary

| Key | Description |
|-----|-------------|
| `n_images` | Total number of images |
| `n_dogs_img` | Number of dog images |
| `n_notdogs_img` | Number of non-dog images |
| `n_match` | Number of label matches |
| `n_correct_dogs` | Correctly classified dog images |
| `n_correct_notdogs` | Correctly classified non-dog images |
| `n_correct_breed` | Correctly classified dog breeds |
| `pct_correct_dogs` | Percentage of correct dog classifications |
| `pct_correct_notdogs` | Percentage of correct non-dog classifications |
| `pct_correct_breed` | Percentage of correct breed classifications |

## Testing with Custom Images

To test the classifier with your own images:

1. Prepare images in JPEG format with approximately square dimensions
2. Name files using the format: `Label_##.jpg`
   - For dogs: `Breed_name_01.jpg` (e.g., `Golden_retriever_01.jpg`)
   - For non-dogs: `Animal_name_01.jpg` (e.g., `Black_bear_01.jpg`)
3. Place images in the `uploaded_images/` directory
4. Run:
```bash
sh run_models_batch_uploaded.sh
```

## Technical Notes

### Image Preprocessing

All images undergo the following preprocessing before classification:
1. Resize to 256 pixels (shortest edge)
2. Center crop to 224x224 pixels
3. Convert to tensor and normalize using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### Label Matching

Pet image labels are extracted from filenames by:
1. Converting to lowercase
2. Splitting on underscores
3. Filtering alphabetic words only
4. Joining with spaces

Classifier labels may contain multiple terms separated by commas (e.g., "dalmatian, coach dog, carriage dog"). A match is determined if the pet label appears anywhere within the classifier label string.

## Limitations

- Similar-looking breeds may be confused (Great Pyrenees/Kuvasz, Beagle/Walker Hound)
- Performance depends on image quality and composition
- Limited to breeds present in ImageNet training data
- Requires images to be named with correct labels for accuracy assessment


