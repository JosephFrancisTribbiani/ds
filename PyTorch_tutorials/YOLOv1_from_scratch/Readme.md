# YOLO v1 loss function

$$
Loss = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} \left [ \left ( x_i - \hat{x_i} \right )^2 + \left ( y_i - \hat{y_i} \right )^2 \right ] + \\
\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} \left [ \left ( \sqrt{w_i} - \sqrt{\hat{w_i}} \right )^2 + \left ( \sqrt{h_i} - \sqrt{\hat{h_i}} \right )^2 \right ] + \\
\sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{obj} \left ( C_i - \hat{C_i} \right )^2 + \\
\lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} 1_{ij}^{noobj} \left ( C_i - \hat{C_i} \right )^2 + \\
\sum_{i=0}^{S^2} 1_{i}^{obj} \sum_{c \in \text{classes}} \left ( p_i(c) - \hat{p}_i(c) \right )^2
$$

Prediction vector by YOLO v1

$$
\left [ \hat{C}_{\hat{b}_1}, \hat{x}_{\hat{b}_1}, \hat{y}_{\hat{b}_1}, \sqrt{\hat{w}_{\hat{b}_1}}, \sqrt{\hat{h}_{\hat{b}_1}}, \ldots, \hat{C}_{\hat{b}_B}, \hat{x}_{\hat{b}_B}, \hat{y}_{\hat{b}_B}, \sqrt{\hat{w}_{\hat{b}_B}}, \sqrt{\hat{h}_{\hat{b}_B}, }
\hat{p}(c_{1}), \ldots, \hat{p}(c_{k}) \right ]
$$

Target vector

$$
\left [ \text{max}_{\hat{b} \in \{\hat{b}_1, \ldots, \hat{b}_B \}} \left ( IoU (b, \hat{b}) \right ), 
x_{b}, y_{b}, \sqrt{w_{b}}, \sqrt{h_{b}}, p(c_1), ..., p(c_k)\right ]
$$

# References

1. [YOLOv1 original paper](https://arxiv.org/pdf/1506.02640.pdf)
2. [Video tutorial](https://www.youtube.com/watch?v=n9_XyCGr-MI)