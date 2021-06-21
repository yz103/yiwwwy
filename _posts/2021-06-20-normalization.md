---

title:  Normalization in Deep Learning
tags: ML/DL
---



Normalization has the advantages to **reduce the training time** and avoid overfitting as a way of **regularization**. 

Challenges during training:

- Choices regarding data preprocessing often make an enormous difference in the final results
- The variables (e.g. affine transformation outputs in MLP) in intermediate layers may take values with widely varying magnitudes and  this drift in the distribution of such variables could hamper the convergence of the network.
- Deeper networks are complex and easily capable of overfitting



Basic idea : 

The normalization scalar $$\sigma$$ can implicitly reduce learning rate and makes learning more stable. 

#### Batch normalization

Batch normalization is a popular and effective technique that consistently accelerates the convergence of deep networks.

Batch normalization is applied to individual layers and works as follows, the inputs are normalized by subtracting their mean and dividing by their standard deviation based on the statistics of the current minibatch. 

Note that **if we tried to apply batch normalization with minibatches of size 1, we would not be able to gain anything**. So the choice of batch size is more important to the performance for the model with batch normalization. 

$$
BN(x)=\gamma\bigodot\frac{x-\hat{\mu_B}}{\hat{\sigma_B}}+\beta
$$

where $\hat{\mu_B}$ is the sample mean and $\hat{\sigma_B}$ is the sample standard deviation of the minibatch $B$

Note that $\gamma$ and $\beta$ are parameters that need to be learned jointly with the other model parameters

$$
\hat{\mu_B} = \frac{1}{|B|}\sum_{x\in B}x​
$$

$$
{\hat{\sigma_B}}^2 = \frac{1}{|B|}\sum_{x\in B}(x-\hat{\mu_B})^2+\epsilon
$$

$$\epsilon$$ is added to avoid dividing by zero.

**Batch normalization in fully connected layer:**

The original paper inserts batch normalization after the affine transformation and before the nonlinear activation function (later applications may insert batch normalization right after activation functions)

The output of the convolutional layer is a 4-dim tensor [B,H,W,C] where B is the batch size, (H,W) is the feature map size, C is the number of channels. For usual batch norm in pseudo-code

```python
# t is the incoming tensor of shape [B, H, W, C]
# mean and stddev are computed along 0 axis and have shape [H, W, C]
mean = mean(t, axis=0)
stddev = stddev(t, axis=0)
for i in 0..B-1:
  out[i,:,:,:] = norm(t[i,:,:,:], mean, stddev)
```

Basically **it computes H\*W\*C means and H\*W\*C standard deviation across B elements**.

**Batch normalization in convolutional layer:**

But the convolutional layer has a special property: filter weights are shared across the input images. That's why it is reasonable to normalize the output in a different way so that each output value take the mean and variance of B*H*W values at different locations. In other words, **there are only C means and standard deviations and each one of them is computed over B\*H\*W values**. The pseudo code is

```python
# t is still the incoming tensor of shape [B, H, W, C]
# but mean and stddev are computed along (0, 1, 2) axes and have just [C] shape
mean = mean(t, axis=(0, 1, 2))
stddev = stddev(t, axis=(0, 1, 2))
for i in 0..B-1, x in 0..H-1, y in 0..W-1:
  out[i,x,y,:] = norm(t[i,x,y,:], mean, stddev)
```

Once the model is trained, we can calculate the means and variances of each layer's variables based on the entire dataset. Thus, batching normalization layers function differently in training mode (normalizing by minibatch statistics) and in prediction mode (normalizing by dataset statistics).

#### Layer normalization

...

#### Weight normalization

...

#### Batch normalization versus layer normalization

- Bath normalization impose the constraint on the size of batch while layer normalization can be used in the case with batch size 1.
- 

To be continued....



Reference:

- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

