# Matthieu DONATI ARRANZ M2 D3S.
# Deep Learning project: N.N.Q. Neural Network Quantization

This project delves into the concept of neural network quantization, specifically focusing on reducing the bit representation of weights in a neural network from 32 bits to 8 bits. The primary case study involves applying these techniques to a ResNet-18 model trained on the Imagenette dataset.

link to notebook: [https://colab.research.google.com/drive/15J-s3Dc5-nA4A7f5zguLfpONLCoEuEjJ?usp=sharing](https://colab.research.google.com/drive/15J-s3Dc5-nA4A7f5zguLfpONLCoEuEjJ#scrollTo=au0emUDxdJbU)

link to failures notebook : https://colab.research.google.com/drive/15J-s3Dc5-nA4A7f5zguLfpONLCoEuEjJ?usp=sharing

### STEPS

#### I Theory
We will try to provide a clear explanation of Quantization, its significance, and its types, including Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).
We will detail the benefits and potential trade-offs of implementing quantization in neural networks.

#### II Resnet18 and Imagenette
We will study the structure of the model Resnet18 and its application to the Imagenette dataset.

#### III Implementing Post-Training Quantization:

Apply Post-Training Quantization to the ResNet-18 model.
Measure and document the model's performance in terms of execution time and accuracy.
Applying Quantization-Aware Training:

#### IV Implement Quantization-Aware Training
Assess and compare the performance metrics with the previous PTQ model.
Advanced Exploration (Optional):


## I Theory
### What is quantization?
#### Context: How are represented number in a CPU ?
Most CPU use 32 or 64 bits. Let's say 32 bits. The computer does not read the number in 10 basis like we do but in a 2 basis, meaning it only uses 0 and 1. 

For most CPU a number is written with the IEEE754 structure in 32  bits ( or 64 bits) system. 

For more information about how it works watch this video : https://www.youtube.com/watch?v=PtFa7LhV96E  
(in French)

But there also exist other model that at one hand are less precise but on the other hand need less memory like the 8bits system. 

#### The problem:
• Most modern deep neural networks are made up of billions of parameters. For example, the smallest LLaMA 2 has 7 billion parameters.  

• When we inference a model (to pass value through the model to get predictionsà, we need to load all its parameters in the memory, this means big models cannot be loaded easily on a standard PC or a smart phone, or just for exemple on Github account with a maximum 25MB files. 

• Just like humans, computers are slow at computing floating-point operations compared to integer operations. Try to do 3 × 6 and compare it to 1.21 × 2.897, which one can you compute faster?

#### The solution:
• Quantization aims to reduce the total amount of bits required to represent each parameter, usually by converting floating-point  numbers into integers. This way, a model that normally occupies 10 GB can be “compressed” to less than 1 GB (depending on the type of quantization used). 
I was able to upload the weights of the model with quantization (11MB) but not the ones of the regular one (44MB).
https://youtu.be/KASuxB3XoYQ?si=k7ElBL6I_8iU2YXW&t=571
During this project we will focus on Quantazing from 32bits to 8bits. 

8 bits format is equivalent to 2^8 different values which is equal to 256 different values. The goal is therefore to map our floating numbers into a set of 256 different integers values that follow each other with equal step. (in fact 255 for symetry purpose).

#### Calibration

For symetric quantization we will quantize from Q (computer Q) to I[ -127 , 127 ]I  
In order to do so we will need to find an alpha value to create a range [-alpha,alpha].
For the first step of the transformation if the input value is outside the range we clip it into it, otherwise it remain unchanged. For the second step just divide by alpha our value, multiply it by 127 and then round it up to the nearest integer. And it is done. 

Exemple: We have the sequence [-0.16, -0.04,  0.07, -0.08, -0.  , -0.09, -0.05, -0.08, -0.07, 0.06]
We could here calibrate it by the bigest absolute value 0.16 in order to avoid clipping and minimising imprecision.
We would obtain [-127,  -32,   56,  -64,   -0,  -71,  -40,  -64,  -56, 48] which is good because if we dequantize it multiply by the scale 0.16/127 we almost lost nothing. But now let's say we add an outlier to to this sequence: 15. Our new alpha would be 15, and our new sequence [-1, -0,  1, -1, -0, -1, -0, -1, -1,  1, 127] where most of its information is lost because of one outlier (mean loss of 0.02 after dequantization). But we could have taken the same scale as when we did not have the value ten, but the value 10 would have become 0.16 after dequantization. There is therefore a tradeoff between the rounding error and the clipping error.

This is why the choice of the calibration parameter is very important. 

To select a good alpha several methods exist: 

- max : maximum absolute value (the one we have seen)
- percentile : We select a certain percentile p of the distribution of the absolute values 
- entropy: chooses an alpha that minimises the KL divergence between the quantized distribution and the original one. (works well for normal distributions)

There are other type of quantization : 
Some quantization method use two parameters alpha and beta it is usefull if the distribution is not symetrical to 0. Process: clip( abs(x/alpha)+beta , 127 , 127)  
Note: Q(0)=beta  so beta was equal to 0 in the previous exemple 
We can modify also the mapping using I[0,255]I instead of I[ -127 , 127 ]I :  clip( abs(x/alpha)+beta' , 0 , 255) usefull a distribution is skewed

#### Post-Training Quantization and Quantization-Aware Training
Post-training quantization consists in quantizing a model after it has been trained. 

Exemple : Let say we have a trained model with 3 imput 2 hidden layers each with 2 nodes with ReLu activation function and 1 output passed into a sigmoid for a probabilty. 
The weight matrix is given by :  

W = [[0.1, -0.3, 0.4], [-1, 2, -0.1]], [[1.2, -0.7], [1, -0.1]], [[3, -4]]]  
and B = [[0.8, -0.6], [-0.5, 0.2], [1]]

For simplicity we will quantize with max value on both parameters. 
The new quantized parameters are :

W_quant = [[[3, -10, 13], [-32, 64, -3]], [[38, -22], [32, -3]], [[95, -127]]]  
B_quant = [[25, -19], [-16, 6], [32]]

For the input we can scale it using the training set for exemple.

Let's say we have an quantized input = [ 1 , 2 , 3 ]

After passing our values through the nodes, and then dequantizing before the sigmoid we obtain 0.11.

Let's suppose that the scale 1 for the input then the same input for the normal neural network. If we pass this input into this neural network we obtain 0.13 therefore for 1-0 loss function we will choose the same outcome.

In our model within each layer we used the exact same scaling rate: 1 for the fist one, 127/4 for the other ones, and the activation function was ReLU where f(tx)=tf(x) for t>0 . Therefore we did not have to dequantize before the end. We just took the value before the final sigmoid and multiplied it by the scale of each layer. We therefore multiplied our value by (1/1)*((4/127)^3)) and then passed the value into the sigmoid.

In some model we might not apply the same scale to each node even from the same layer. If it is the case we cannot wait until the end to dequantize. It must be done before the next layer otherwise values at different same scale might interact at the next layer which might leed to problems.


#### Different ways to apply quantization
 There are three main different way to apply quantization:

- Naive Quantization :
All operators are quantized  and  calibrated using the same method.

This technique is very fast but not always efficent. Some layers in model can be very sensitive and it might reduce drastically the precision of the model to quantize them. 

We also have cases where some operator cannot be transformed in INT8, it might even worseb not only the accuracy but also the latency.

- Hybrid Quantization :

Some operators are quantized others are not. It necessissate to know well the structure of the model, to know which layers are incompatible or sensitive and which are less. A way to do that is an equivalent to a grid search or that will test different model where parameters are quantized or not and evaluate the sensivity (the difference in performance)


- Selective Quantization: The most sophiscated one, it can selecte different calibrate at different granularity (level), by tensor or by channel for exemple in case of convolution.
We can also additonaly change  parts of the model to better suit it.

####  Quantization-Aware Training 
Quantization aware training consists in adding layer of quantization directly followed by dequantization in the initial model between steps during the training. This way the model get more use to these less precise values. Once the model trained, we apply Post Training Quantization.



#### Application
We will try to apply quantization to a ResNet18 model. This model is used alot for image recogniction, especially on the Imagenet dataset that has 1000 label. This model uses Convolution, Pooling, Average Pooling, Neurons, and residual blocks. This model was created to counter the vanishing/exploding gradient problem. Deep model tend to have graident that explode or vanish during forward and backward propagation because of the huge number of multiplications. After a certain amound of hidden layers models tend to lose in accuaracy on their training test, which is not logical in theory.(If we take a random with 1 mean values, and we multiply each value drawn, the probability that is vanishes is great or explode is great, even if it is a Martingale. There is also the impact of computer approximation repeted many times).           
The ResNet model (18,50... depending on the number of layers), will do the same as a model with convolution, pooling etc... ( which is already a bit sophiscated), but during some steps, values will be copied and then the model will continue for two layers and then the copied value is added to the two-layers-later value.

We will use this model on a the Imagenette dataset, a subset of Imagenette only 10 labels and 13000 pictures. 
Here is a video with great visuals:
https://www.youtube.com/watch?v=o_3mboe1jYI



####
For the application I used a model of Resnet18 pre-constructed by PyTorch without pretrained weights as they are all trained on . The package fastai allow us to download Imagenette.

I first had to modify the Resnet18 model's structure. Indeed we had to go from 1000 to 10 labels. 
The second step was to train the model. 
- The Imagenette dataset was separated between train and validtion pictures. 
- The model was then trained on the training set, using the ADAM optimiser with a learning rate of 0.1. The training and testing set where devided in 148 of size 64 and  62 batches of size 64 repesctively. I trained for 5 epochs. Each epoch was taking a little bit more than 40 mininutes for a total of under 4 hours. 

After training for 5 epochs my function the accuaracy score on the test set was around 68%. I tries 2 more epochs, while the training loss was lowering the test set score was also lowering. It proabably meant overfitting of the training set but not for sure because I unfortunately did not graph the accuaracy to check the trends. 

I tried several technique to apply quantization but unfortunately it did not work properly ... until it did.

Very few people had quantized a ResNet model online and the advice on blogs did not really seem to work. Same thing for ChatGpt. On youtube  ( Umar Jamil https://www.youtube.com/watch?v=0VdNflU08yA ) applied a Post Training quantification to a different model he had created but the code he used looked universal, applicatble to any model. 
The code places observers at each steps (node or activation function) of the model, the "observers" observe the range of the data passing through, and in a third step,apply the quantization of each weights bases on the range of these values. 

In the video (following link for the code), the weights are transformed into integers whereas when I applied the code the weight remained as float in apparence. Neverless the value where stored as torch.qint8 bits. The size of the model had dropped by a factor of 4. But when I tried to apply the model apparentely there was a "backend problem " due to a lack of compability.  Umar Jamil had not encountered (maybe due to the simplicity of his model).

It then tried to quantize by hand using quantiles 0.01 0.99 as beta and alpha quantile and the median as zero-point. I then multiplied by 255, rounded the values and clip it into -127, 127, The value were now in theory quantize but not practically because there integers where stock at the 32 bits format and wouldn't accept any other. 

I tried a last step that consisted in using a pytorch quantified model. This class could have been very usefull but the it was design for the Imagenette model. But this class could only quantize with weights (PTQ) the orginal model, that has 1000 outputs. But the weights of this model had the same structure as the the one I created based on the Umar Jamil video.  I therefore imported them without the weights, and then assigned the weights from the Umar Jamill model to this one. 

The results were terrible, 9%. It would only assign 0 values which apparentely represented 9% of the value of the test set ("trench").

On the positive side, I was able to oberseve the pace of the model, iteration on the training took half the time approximately

I also tried other codes but they did not work at all, therefore they are not worth quoting.

I was about to give up after hours of looking for solution but just before I did and ( way pass the deadline limit ), and I reread the name of the pytorch class I used torchvision.models.quantization.resnet18, the module was called "QuantizableResnet" as if Pytorch had created on function Resnet that could not be Quantized and another one that could. 

And when I applied the same process as in the Umar Jamill Video it now worked perfectly.

I just had to use this torchvision.models.quantization.resnet18 and set quantization to false, change the last layer, aplly to it the trained weights, put observers, test the model, quantize with observertions and done.

The results where quite impressive:
2.2 times faster, 25.24% of the size ( as before ) but for a loss of only 0.43% of the accuaracy

#### Post Aware Quantization.
After applying post aware Quantization the model accuaracy drop significantally (around %) but it might have been because of overfitting, as I said before, when I add run one more epoch, the precision of the model on the testing had lowered. When I tested normal the model after the new epoch, the model has indeed lost 5% of accuaracy with the New Epoch. Which mean, that the gap is now much bigger than before between the quantize and not quantize model. 
PAQ was not sucessfull here.

Note: In the Umar Jamill video the accuaracy drops a little after applying PAQ. 









### Unquoted source:
https://deci.ai/quantization-and-quantization-aware-training/
https://pytorch.org/tutorials/recipes/quantization.html
https://pytorch.org/docs/master/quantization.html#quantization-aware-training
