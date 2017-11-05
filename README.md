# Gradient Regularized GAN
Code for the regularizer proposed in the NIPS 2017 paper on "Gradient descent GAN optimization is locally stable"
<a href="https://arxiv.org/abs/1706.04156">arXiv:1706.04156</a>

The code for the Gaussian experiments is based on https://github.com/poolio/unrolled_gan but modified to reflect the parameters specified in the Unrolled GANs paper <a href="https://arxiv.org/abs/1611.02163">arxiv:1611.02163</a>. The experiments for our paper were performed on Tensorflow 1.1.0 and Keras 1.2. 

Run ```python <filename>.py``` where ```<filename>``` is either ```gaussian-toy-unrolled.py``` or ```gaussian-toy-regularized.py``` to run the unrolled GAN and the gradient-norm-regularized GAN respectively.


