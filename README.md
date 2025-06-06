# Retinal-Vessel-Segmentation-Using-Deep-Learning
![ alt text ](https://img.shields.io/badge/license-MIT-green?style=&logo=)
![ alt text ](https://img.shields.io/badge/-Jupyter-F37626?logo=Jupyter&logoColor=white)
![ alt text ](https://img.shields.io/badge/-NumPy-013243?logo=Numpy&logoColor=white)
![ alt text ](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=TensorFlow&logoColor=white)
![ alt text ](https://img.shields.io/badge/-Keras-D00000?logo=Keras&logoColor=white)

The motivation behind this project stems from the increasing need for automated systems that can assist healthcare professionals in diagnosing retinal diseases with greater precision and efficiency. Manual segmentation of retinal vessels is time-consuming, error-prone, and highly dependent on the skill of the clinician. The attached notebook demonstrates an example of how such a system can be implemented.

The backbone, the underlying neural network architecture used for feature extraction, was the EfficientNetB0. For a binary classificaton the activation function was sigmoid. The loss function was focal Dice loss, which unifies both focal cross-entropy loss and Dice loss. This loss function is designed to handle class imbalance and asymmetric losses. We can express the loss (its components) with following formulas:
1. Dice loss:

$$\text{Dice Loss} = 1 - \frac{2 \Sigma{(p \cdot y)} + \epsilon}{\Sigma{p} + \Sigma{y} + \epsilon}$$

where:
* $p$ - predicted probability (after sigmoid)
* $y$ - ground truth label (0 or 1)
* $\epsilon$ - small constant to avoid division by zero (e.g. 1e-6 or 1e-7)

2. Binary focal loss:

$$\text{Focal Loss} = -\alpha(1 - p)^{\gamma}y log(p + \epsilon) - (1 - \alpha)p^{\gamma} (1 - y) log(1 - p + \epsilon)$$

where:
* $\alpha \in [0, 1]$ - weight balancing factor for class 1, the weight for class is $1.0 - \alpha$ (e.g. 0.25)
* $\gamma \ge 0$ - focusing parameter used to modulate focal factor (e.g. 2.0)

3. Focal Dice loss:

$$\text{Focal Dice Loss} = \text{Focal Loss} + \text{Dice Loss}$$

or optionally, with a balancing factor:

$$\text{Focal Dice Loss} = \lambda \cdot \text{Focal Loss} + (1 - \lambda) \cdot \text{Dice Loss}$$

where:

* $\lambda \in [0, 1]$ - balances the contribution of focal Loss and Dice Loss

The model's optimizer was Adam. The learning rate was set to 1e-4 for 210 epochs. Below are the learning curves for each chosen evaluation metric for monitoring U-Net's learning:
<div align='center'>
<img src='https://github.com/user-attachments/assets/6d2985a8-d38c-478f-8339-01213db643d7' width='500'/>
</div>

The U-Net model after evaluation on testing data achieved the following results:
| Accuracy | F1 Score | IoU    | Loss   | MCC    | Precision | Recall | Specificity |
|----------|----------|--------|--------|--------|-----------|--------|-------------|
| 0.9627   | 0.7672   | 0.7928 | 0.3726 | 0.7494 | 0.8275    | 0.7191 | 0.9857      |

The model demonstrates robust and reliable performance, with balanced metrics that suggest it generalizes well across different images. High specificity and precision make the model trustworthy in not generating false alarms. This is important in sensitive medical diagnostics. While recall leaves some room for refinement, especially in certain edge cases, the overall segmentation quality remains strong. A few examples of ground truth masks and their predictions:<br>
<div align='center'>
<img src='https://github.com/user-attachments/assets/aabb4105-fe25-40bc-ae5e-a7f3de047e5c' width='400'/><br>
</div>
<div align='center'>
<img src='https://github.com/user-attachments/assets/342c418e-ba4e-4b8e-adce-6a9f01d8e1cd' width='400'/><br>
</div>
<div align='center'>
<img src='https://github.com/user-attachments/assets/858546e2-a425-4722-ad12-21a14a52688c' width='400'/>
</div>

While the results are promising, particularly in segmenting fine vessel structures, there remains potential for further improvement. Future work may focus on refining microvessel detection, addressing class imbalance more effectively, and exploring attention mechanisms and different architectures to further boost performance and generalizability in real-world clinical settings.
