# MFMS-Swin
Interpretable MFMS-Swin for diabetic retinopathy grading with attention mask regularization
# Interpretable MFMS-Swin

**Diabetic Retinopathy Grading Based on Multi-Scale Feature Fusion and Attention Mask Regularization**

Interpretable MFMS-Swin is a diabetic retinopathy (DR) grading framework built on the Swin-Tiny Transformer. It combines multi-scale feature fusion, spatial and channel attention, and attention-mask learning guided by auxiliary classification supervision and sparsity regularization. The framework produces DR grading predictions and learned spatial attention maps for visual interpretation.

This README documents the supplied implementation in `xunlian(4).py`. The commands below use that filename; update them if you rename the script in your repository.

## Method

The framework contains three main components:

1. **Multi-scale feature learning.** Swin-Tiny extracts features from four hierarchical stages. The Multi-scale Feature Fusion and Dual Attention Enhancement (MFMS) module aligns their channel dimensions and spatial resolutions, fuses them, and applies spatial attention and channel attention with feature gating.
2. **Attention-mask learning.** An attention generator produces a single-channel mask from the fused features. The mask is upsampled to the input resolution and multiplied with the preprocessed, normalized input tensor. During training, the masked input is passed through the same feature extractor and classifier, with supervision from the same grading targets.
3. **Ordinal Focal Loss.** Class-weighted focal loss is combined with an ordinal penalty based on the expected predicted grade. The training objective also includes the masked-input classification loss and a sparsity penalty on the mean mask activation.

For a single input view, inference returns the primary prediction and can also return its attention mask without classifying the masked input again. Four-view test-time augmentation (TTA) evaluates the original image, horizontal flip, vertical flip, and 90-degree rotation, then averages their class probabilities.

## Reported Results

The following results are reported in the manuscript using four-view TTA.

| Dataset | Accuracy | Quadratic Weighted Kappa | Weighted F1-score |
| --- | ---: | ---: | ---: |
| APTOS 2019 | 0.907 | 0.939 | 0.903 |
| DDR | 0.901 | 0.934 | 0.900 |

Reproduction requires the corresponding data splits, preprocessing protocol, initialization weights, and software environment. These values are manuscript results, rather than benchmarks generated automatically by this README.

## Environment

The training script uses:

- PyTorch and torchvision
- timm
- NumPy and pandas
- scikit-learn
- Pillow and OpenCV
- Matplotlib and seaborn

Install a compatible PyTorch/torchvision pair using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), choosing the build for your operating system and CUDA environment. Then install the remaining dependencies:

```bash
python -m pip install timm numpy pandas scikit-learn Pillow opencv-python matplotlib seaborn
```

An NVIDIA GPU is recommended for training. The installation command above lists dependencies; it does not specify the exact environment used for the manuscript experiments. Record the versions from that environment when preparing a reproducible release.

The MFMS implementation expects Swin feature maps in NHWC format and converts them to NCHW. Keep the `timm` version consistent with this interface and the pretrained checkpoint format. See the [timm documentation](https://huggingface.co/docs/timm/index).

## Datasets

Obtain the datasets from their original sources:

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- [DDR dataset](https://github.com/nkicsl/DDR-dataset)

Each label CSV must contain these columns:

```csv
id_code,diagnosis
image_001,0
image_002,2
image_003,4
```

`id_code` identifies an image in the corresponding image directory. Filenames can include an extension; the loader also tries `.png` and `.jpg` when an extension is omitted.

| Label | DR grade |
| ---: | --- |
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | PDR |

Use the manuscript's preprocessing protocol and save the processed RGB images at **384 × 384 pixels**. The supplied training and evaluation transforms apply ImageNet normalization but do not resize images.

With the supplied configuration, the following paths are expected under `CONFIG['root_dir']`:

| Relative path | Contents |
| --- | --- |
| `train_1.csv` | Training image identifiers and grading labels |
| `valid.csv` | Validation image identifiers and grading labels |
| `test.csv` | Test image identifiers and grading labels |
| `train_images_processed/` | Processed training images |
| `val_images_processed/` | Processed validation images |
| `test_images_processed/` | Processed test images |

Keep training, validation, and test partitions separate. Apply any offline augmentation or oversampling only to the training partition.

If you also use the supplied `huafenshuju.py` preparation script, adapt its outputs to the loader: it writes `image` and `grade` columns, which must be renamed to `id_code` and `diagnosis`. Its CSV names and image directories also differ from the training defaults.

Check that every image listed in the CSVs exists and can be opened. The supplied loader replaces an unreadable image with a blank image, so input validation is needed before training or evaluation.

## Configuration and Training

Edit the `CONFIG` dictionary at the top of the training script. In particular, set `root_dir` and `pretrained_path` to your actual local paths. For example, place these assignments immediately after the existing `CONFIG` dictionary:

```python
CONFIG['root_dir'] = './data/aptos2019'
CONFIG['pretrained_path'] = './weights/swin_tiny_patch4_window7_224.pth'
```

Use a separate dataset root for each dataset. Update the CSV and image-directory entries if your prepared data use different names.

The script initializes the backbone with `pretrained=False` and loads a local checkpoint through `pretrained_path`. Setting the constructor's `pretrained=True` argument alone does not enable automatic weight downloading in this implementation. Confirm that the intended checkpoint is present and its weights are compatible with the backbone.

Default training settings are:

| Setting | Value |
| --- | --- |
| Backbone | `swin_tiny_patch4_window7_224` |
| Input resolution | 384 × 384 |
| Number of classes | 5 |
| Fused feature channels | 256 |
| Batch size | 32 |
| Epochs | 60 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.01 |
| Focal gamma | 2.0 |
| Ordinal penalty weight | 0.1 |
| Main-loss warm-up | First 25 epochs |
| Maximum masked-input classification weight | 0.1 |
| Mask sparsity weight | 0.03 |
| EMA decay | 0.999 |

After the first 25 epochs, the mask sparsity penalty is included and the masked-input classification weight gradually increases from zero to 0.1 over 20 epoch increments. The configuration name `cons_weight` controls this auxiliary classification weight.

The script also uses inverse-fourth-root class-frequency weights, Mixup, label smoothing, image augmentation, mixed-precision training, and a cosine learning-rate scheduler with warm restarts.

Run training from the directory containing the script:

```bash
python "xunlian(4).py"
```

The supplied script is configured through `CONFIG`; it does not implement a command-line argument parser. It selects the best EMA checkpoint by validation QWK and then evaluates that checkpoint on the validation and test partitions using four-view TTA.

## Saved Outputs

Outputs are written under `CONFIG['root_dir']`:

| Output | Description |
| --- | --- |
| `best_interpretable_model.pth` | Best EMA model selected by validation QWK |
| `val_prediction_results.csv` | Validation predictions and grade-error distances |
| `test_prediction_results.csv` | Test predictions and grade-error distances |
| `val_confusion_matrix.png` | Validation confusion matrix |
| `test_confusion_matrix.png` | Test confusion matrix |
| `val_classification_report.txt` | Validation classification report, including macro and weighted averages |
| `training_curves.png` | Training and validation curves |
| `vis_results/` | Attention maps and overlays for samples from the validation loader |

The script prints test accuracy and QWK. To calculate test weighted F1 from its exported predictions, run:

```python
import pandas as pd
from sklearn.metrics import f1_score

predictions = pd.read_csv('./data/aptos2019/test_prediction_results.csv')
weighted_f1 = f1_score(
    predictions['true_label'],
    predictions['pred_label'],
    average='weighted',
)
print(f'Test weighted F1: {weighted_f1:.4f}')
```

Replace the CSV path with the output path for your run.

## Prediction and Attention Masks

The following example loads a trained checkpoint and returns a prediction and attention mask for one preprocessed image:

```python
import importlib.util
from pathlib import Path

import torch
from PIL import Image

script_path = Path('xunlian(4).py')
spec = importlib.util.spec_from_file_location('mfms_impl', script_path)
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
implementation.CONFIG['img_size'] = 384

model = implementation.InterpretableMFMSSwin(
    num_classes=5,
    pretrained=False,
    pretrained_path=None,
).to(device)

checkpoint_path = Path('./data/aptos2019/best_interpretable_model.pth')
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

image = Image.open('path/to/preprocessed_image.png').convert('RGB')
x = implementation.val_test_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits, attention_mask = model(x, return_map=True)
    probabilities = torch.softmax(logits, dim=1)
    predicted_grade = probabilities.argmax(dim=1).item()

print('Predicted DR grade:', predicted_grade)
print('Attention mask shape:', tuple(attention_mask.shape))
```

Update the script, checkpoint, and image paths as needed. The input image must already have the expected preprocessing and resolution. This example uses one input view; the manuscript metrics use four-view TTA.

The `visualize_heatmap` function generates attention maps and image overlays. For display, masks are min–max normalized per image, a circular display mask is applied, and the color heatmap is blended with the fundus image using equal weights. These display operations do not modify the mask used for training.

## Citation

The associated manuscript is titled **Diabetic Retinopathy Grading Based on Multi-Scale Feature Fusion and Attention Mask Regularization**. A complete citation can be added when its public bibliographic record is available.

## Acknowledgments

This work uses the Swin Transformer architecture, the `timm` library, and the APTOS 2019 and DDR datasets. Please also acknowledge their original authors when using these resources.

- [Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)
- [timm](https://github.com/huggingface/pytorch-image-models)
