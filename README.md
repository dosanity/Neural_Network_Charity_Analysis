# Neural Network Charity Analysis

## Project Overview

We will be analyzing data to create a binary classifier that is capable of determining whether certain organizations will be successful if funded by a philanthropic foundation. We will be utilizing machine learning algorithms and neural networks to apply features on a provided dataset. This will help determine which organizations will receive donations.

Neural networks are advanced machine learning techniques that are modeled after neurons in the brain that are effective with complex datasets. These networks are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Deep neural networks are used for analyzing images and natural language processing datasets since they are effective at detecting complex, nonlinear relationships.

## Resources
+ Analysis Software: `Python 3.10`, `Jupyter Lab 3.4.4`
+ Data Source: `charity_data.csv`

## Results

### Data Preprocessing

+ The target variable in our model was `IS_SUCCESSFUL`.
+ The feature variables in our model are the remaining variables. (`STATUS`, `ASK_AMT`, `APPLICATION TYPE`, `AFFILIATION`, etc.)
+ `EIN` and `NAME` columns were removed since these variables are neither targets nor feautures.

### Compiling, Training, and Evaluating

After the data was preprocessed, we used the following parameters to compile, train, and evaluate the model:

+ Initial Model: 5981 parameters = First Hidden Layer + Second Hidden Layer + Output Layer
+ First Hidden Layer: 3520 parameters = [43 inputs (from input layer) * 80 neurons] + (80 bias terms)
+ Second Hidden Layer: 2430 params = [80 inputs (from first hidden layer) * 30 neurons] + (30 bias terms)
+ Output Layer: 31 params = [30 inputs (from second hidden layer) * 1 neuron] + (1 bias term)

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 80)                3520      
                                                                 
 dense_1 (Dense)             (None, 30)                2430      
                                                                 
 dense_2 (Dense)             (None, 1)                 31        
                                                                 
=================================================================
Total params: 5,981
Trainable params: 5,981
Non-trainable params: 0
_________________________________________________________________
```

The target performance for accuracy rate is greater than 75%. The model only achieved an accuracy rate of 72.7%.

> Python Code:

```
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

> Output:

```
268/268 - 1s - loss: 0.5956 - accuracy: 0.7271 - 609ms/epoch - 2ms/step
Loss: 0.5956281423568726, Accuracy: 0.7271137237548828
```

### Optimizing Model Performance




