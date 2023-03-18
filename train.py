# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:15:41 2023

@author: SHREYASH
"""
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
import warnings
import argparse


def dataset_preprocess(dataset):
  #change the code below to accept different dataset
  if dataset=="mnist":
   (X_1, Y_1), (X_test, Y_test) = mnist.load_data()
  else:
   (X_1, Y_1), (X_test, Y_test) = fashion_mnist.load_data()  
  #importing dataset again and normalizing
  X_1 = X_1.reshape(X_1.shape[0],-1)/255.0
  X_test = X_test.reshape(X_test.shape[0],-1)/255.0

  #training and validation split as specified in the question 10%
  X_train, X_val, Y_train, Y_val= train_test_split(X_1,Y_1,test_size=0.1,random_state=0)
  
  #one hot encoding
  Y_train_encoded=np.zeros((Y_train.shape[0],10))
  for i in range(len(Y_train)):
    Y_train_encoded[i][Y_train[i]]=1

  Y_val_encoded=np.zeros((Y_val.shape[0],10))
  for i in range(len(Y_val)):
    Y_val_encoded[i][Y_val[i]]=1

  Y_test_encoded=np.zeros((Y_test.shape[0],10))
  for i in range(len(Y_test)):
    Y_test_encoded[i][Y_test[i]]=1

  return X_train.T,X_test.T,X_val.T,Y_train.T,Y_val.T,Y_test.T,Y_train_encoded.T,Y_val_encoded.T,Y_test_encoded.T

#---------------------------------------------------------------ACTIVATION FUNCTIONS AND THEIR GRADIENTS-------------------------------------------------------------------------------
def relu(X):
  return np.maximum(0,X)

def grad_relu(X):
  return X>0

def sigmoid(X):
  return 1/(1+np.exp(-X))

def grad_sigmoid(X):
  return (sigmoid(X))*(1-sigmoid(X))

def tanh(X):
  return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def grad_tanh(X):
  return 1-(tanh(X))**2

def softmax(X):
  e_X = np.exp(X - np.max(X, axis = 0))
  return e_X / e_X.sum(axis = 0)


activation_function={
      'sigmoid':sigmoid,
      'relu':relu,
      'tanh':tanh
}

grad_activation_function={
      'sigmoid':grad_sigmoid,
      'relu':grad_relu,
      'tanh':grad_tanh
}
#---------------------------------------------------------------------------LOSS FUNCTION---------------------------------------------------------------------------------------------
def cross_entropy_loss(Y_encoded,Y,Y_pred,lambd,b):
 loss = (-np.sum(np.multiply(Y_encoded,np.log(Y_pred)))+((lambd/2.)*b))/Y.shape[0]
 return loss
 
def squared_error_loss(Y_encoded,Y,Y_pred,lambd,b):
  loss=((1/2)*(np.sum(Y_encoded-Y_pred))**2)/Y.shape[0]+(lambd*b*0)
  return loss

loss_function={
       'cross_entropy':cross_entropy_loss,
       'square_loss':squared_error_loss
}   


def grad_cross_entropy(Y,Y_pred):
  return (Y_pred-Y)

def grad_squared_error_loss(Y,Y_pred):  
  return (Y_pred-Y)*(Y_pred)*(1-Y_pred)

grad_loss_function={
    'cross_entropy':grad_cross_entropy,
    'square_loss':grad_squared_error_loss
}

#----------------------------------------------------------------------INITIALISATION FUNCTIONS---------------------------------------------------------------------------------------
def random_initialisation(shape):
  # Initialising a random matrix with given dimensions (shape) as tuple
  np.random.seed(0)
  return np.random.randn(*shape)*0.3 #multiplied by 0.5 to have smaller values, to have better initialisation weights

def xavier_initialisation(shape):
    # Calculate the number of inputs and outputs
    n_in = shape[0]
    n_out = shape[1]    
    # Calculate the variance of the normal distribution
    variance = 2.0 / (n_in + n_out)
    # Initialize the weights with random values drawn from the normal distribution
    np.random.seed(0)
    weights = np.random.randn(n_in, n_out) * np.sqrt(variance)*2.0
    
    return weights

initialisation_function={
    'random':random_initialisation,
    'xavier':xavier_initialisation
}
#------------------------------------------------------------------INITIALIZING WEIGHTS AND BIASES------------------------------------------------------------------------------------


def initialize_w_b(input_layer,hidden_layer,output_layer,init):
  weights=[]
  biases=[]
  layers=[input_layer]+hidden_layer+[output_layer]
  for i in range(len(hidden_layer)+1): 
    weights.append(initialisation_function[init]((layers[i+1],layers[i])))
    biases.append(np.random.randn(layers[i+1],1)*0.3)
  return weights, biases

class NeuralNet():

  def __init__(self,input_layer,hidden_layer,output_layer,initialisation_func,act_function,loss_func,dropout_rate):
     self.input_layer=input_layer
     self.hidden_layer=hidden_layer
     self.output_layer=output_layer
     self.initialisation_func=initialisation_func
     self.act_function=act_function
     self.loss_func=loss_func
     self.weights,self.biases = initialize_w_b(self.input_layer,self.hidden_layer,self.output_layer,self.initialisation_func)
     self.layer_size=len(self.hidden_layer)
     self.dropout_rate=dropout_rate


  
  def forward_propagation(self,X):
     #pre-activation
     self.a=[]
     #post-activation
     self.h=[]
     self.D=[]
     l=0
     
     #pre-activation and post-activation for input layer and first hidden layer
     self.a.append((self.weights[l]@X)+self.biases[l])#WX+b
     if(self.dropout_rate!=0):
                    dropRate=(1-self.dropout_rate)
                    d= np.random.rand(self.a[l].shape[0], self.a[l].shape[1])
                    d=d<dropRate
                    self.D.append(d)
                    self.a[l]=self.a[l]*d
                    self.a[l]=self.a[l]/dropRate
     self.h.append(activation_function[self.act_function](self.a[l]))
     
     #pre-activation and post-activation between hidden layers
     for l in range(1,self.layer_size):
       self.a.append((self.weights[l]@self.h[l-1])+self.biases[l])
       if(self.dropout_rate!=0):
                    dropRate=(1-self.dropout_rate)
                    d= np.random.rand(self.a[l].shape[0], self.a[l].shape[1])
                    d=d<dropRate
                    self.D.append(d)
                    self.a[l]=self.a[l]*d
                    self.a[l]=self.a[l]/dropRate
       self.h.append(activation_function[self.act_function](self.a[l]))
       
     #pre-activation and post-activation between last hidden layer and output layer
     l=self.layer_size 
     self.a.append((self.weights[l]@self.h[l-1])+self.biases[l])
     self.h.append(softmax(self.a[l]))
     
     return self.h[-1]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------Q3 BACK PROPPAGATION FRAMEWORK WITH OPTIMIZATION FUNCTIONS-----------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
  def back_propagation(self,X,Y):

        g_a  = [0]*(self.layer_size+1)
        g_h  = [0]*(self.layer_size+1)
        g_w  = [0]*(len(self.weights))
        g_b  = [0]*(len(self.biases))
        batch_size = X.shape[1] 

        for k in reversed(range(self.layer_size+1)):
            #calculating loss function gradients for weights and biases at output
            if k == self.layer_size:
              g_a[k]=grad_loss_function[self.loss_func](Y,self.h[k])
            
            #calculating gradients for hidden layers     
            else:
                g_h[k] = (1/batch_size)*((self.weights[k+1].T)@(g_a[k+1]))
                g_a[k] = (1/batch_size)*((g_h[k])*(grad_activation_function[self.act_function](self.a[k])))#Here we use '*' operator for elementwise matrix multiplication
                if(self.dropout_rate!=0):
                    g_a[k]=g_a[k]* self.D[k-1]
                    g_a[k]=g_a[k]/(1-self.dropout_rate)
                
            #calculating gradients of weights 
            if k == 0:
                g_w[k] = (1/batch_size)*((g_a[k])@(X.T)) 
            else:
                g_w[k] = (1/batch_size)*((g_a[k])@(self.h[k-1].T))
            
            #calculating gradients of biases
            g_b[k]  = (1/batch_size)*np.sum(g_a[k], axis=1, keepdims = True)
        return g_w,g_b
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------STOCHASTIC GRADIENT DESCENT AND OTHER OPTIMIZERS-------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
  def stochastic_gradient_descent(self,W,b,g_w,g_b,lr,lambd):
      #Weights=Weights-(learning rate)*(grad weights)-(learning rate*lambda)*(Weights)(this is weight decay with L2 regularization)
      W=W-np.multiply(lr,g_w)-np.multiply(lr*lambd,W)

      #biases=biases-(learning rate)*(grad biases)
      b=b-np.multiply(lr,g_b)
      return W,b
  
  def momentum_based_gradient_descent(self,W,b,g_w,g_b, u_w_i, u_b_i, lr, lambd, gamma):
      #u_t=(beta)*(u_t-1)+(grad weight)[u_0 is initialised as zero]
      u_w=np.multiply(gamma,u_w_i)+g_w
      #weights=weights-(lr)*(updated_weights)-(weight decay L2 regularization)
      W=W-np.multiply(lr,u_w)-np.multiply(lr*lambd,W)
      
      #similarly for biases but without weight decay term
      u_b=np.multiply(gamma,u_b_i)+g_b
      b=b-np.multiply(lr,u_b)
      return W,b,u_w,u_b
  
  def nesterov_accelerated_gradient_descent(self,W,b,g_w,g_b,lr,lambd,gamma,u_w_i, u_b_i,b_input,b_output):
      w_t=W
      b_t=b
      #here we make changes to global variables as we need to update the weights(look ahead) for calculating gradients
      self.weights = self.weights - np.multiply(gamma,u_w_i)
      self.biases = self.biases - np.multiply(gamma,u_b_i)
      output =  self.forward_propagation(b_input)
      #in this step, gradients are recalculated in global variables with updated weight values 
      g_weights,g_biases = self.back_propagation(b_input,b_output)

      #u_t=(gamma)*(u_(t-1))+gradient of(w_t-(gamma)*(u_(t-1)))
      u_w=np.multiply(gamma,u_w_i)+g_weights
      #weights=weights-(lr)*(updated_weights)-(weight decay L2 regularization)
      w_t = w_t - u_w - np.multiply(lr*lambd,w_t)

      #similarly for biases but without weight decay term
      u_b=np.multiply(gamma,u_b_i)+g_b
      b=b-np.multiply(lr,u_b) 

      return w_t,b,u_w,u_b
  
  def rmsprop(self,W,b,g_w,g_b,lr,lambd,beta,eps,vw,vb):
      #V_t= (beta)*(V_t-1)+(1-beta)*(grad weight)^2 
      vw = np.multiply(vw,beta) + np.multiply(1-beta,np.power(g_w,2))           
      #similarly for biases
      vb = np.multiply(vb,beta) + np.multiply(1-beta,np.power(g_b,2))
      
      #w_t= (w_t-1)-((lr)/(V_t+eps)^(1/2))*(grad weight)-(L2 regularization weight decay)
      W = W - np.multiply(g_w,lr/np.power(vw+eps,1/2))- np.multiply(lr*lambd,W)
      #similarly for biases
      b = b - np.multiply(g_b,lr/np.power(vb+eps,1/2))
      return W,b,vw,vb
  
  def adam(self,W,b,g_w,g_b,beta1,beta2,lr ,m_t_i ,v_t_i ,m_b_i ,v_b_i,eps,i,lambd):
      
      m_t = np.multiply(beta1,m_t_i) + np.multiply(1-beta1,g_w)
      v_t = np.multiply(beta2,v_t_i) + np.multiply(1-beta2,np.power(g_w,2))
      m_b = np.multiply(beta1,m_b_i) + np.multiply(1-beta1,g_b)
      v_b = np.multiply(beta2,v_b_i) + np.multiply(1-beta2,np.power(g_b,2))
                
      #normalization of moment          
      m_hat_w = m_t/(1 - np.power(beta1,i+1))
      m_hat_b = m_b/(1 - np.power(beta1,i+1))

      #normalization          
      v_hat_w = v_t/(1 - np.power(beta2,i+1))
      v_hat_b = v_b/(1 - np.power(beta2,i+1))
      
      W = W - ((lr / np.power(v_hat_w + eps, 1/2)) * m_hat_w) - np.multiply(lr*lambd,W)
      
      b = b - ((lr / np.power(v_hat_b + eps, 1/2)) * m_hat_b)
      return W,b,m_t,v_t,m_b,v_b
  
  def nadam(self,W,b,g_w,g_b,beta1,beta2,lr ,m_t_i ,v_t_i ,m_b_i ,v_b_i,eps,i,lambd):
      m_t =  np.multiply(beta1,m_t_i) + np.multiply(1 - beta1,g_w)
      v_t =  np.multiply(beta2,v_t_i) + np.multiply(1 - beta2,np.power(g_w, 2))

      m_b =  np.multiply(beta1,m_b_i) + np.multiply(1 - beta1,g_b)
      v_b =  np.multiply(beta2,v_b_i) + np.multiply(1 - beta2,np.power(g_b, 2))
                
      m_hat_w = m_t / (1 - np.power(beta1, i+1)) 
      v_hat_t = v_t / (1 - np.power(beta2, i+1))

      m_hat_b = m_b / (1 - np.power(beta1, i+1)) 
      v_hat_b = v_b / (1 - np.power(beta2, i+1))
  
      a1 = (1-beta1)/(1-np.power(beta1,i+1))
      update_w = np.multiply(lr/(np.power(v_hat_t + eps,1/2)),(np.multiply(a1,g_w) + np.multiply(beta1,m_hat_w)))
      update_b = np.multiply(lr/(np.power(v_hat_b + eps,1/2)),(np.multiply(a1,g_b)+np.multiply(beta1,m_hat_b) ))
      W = W - update_w - np.multiply(lr*lambd,W)
      b = b - update_b    
      return W,b,m_t,v_t,m_b,v_b
  

  def predict(self, X,Y ):
      output =  self.forward_propagation(X)
      out_class=(np.argmax(output,axis=0))
      accuracy = round(self.accuracy_score(X, Y))
      return accuracy , out_class
  
  def accuracy_score(self, X, Y):
    pred_labels = np.argmax(self.forward_propagation(X), axis=0)
    return 100*(np.sum(pred_labels == Y) / len(Y))

  
  def predict_one_hot_encoded(self, X,Y ):
      output =  self.forward_propagation(X)
      accuracy = round(self.accuracy_score(X, Y))
      return output,accuracy


  def train(self,X_train,y_train,X_val ,y_val ,learning_rate,epochs, optimiser='gd',batch_size = 64,lambd=0.0005,WandB=False): 

      update_w = np.zeros(np.array(self.weights).shape)
      update_b = np.zeros(np.array(self.biases).shape)
      update_w_i = np.zeros(np.array(self.weights).shape)
      update_b_i = np.zeros(np.array(self.biases).shape)
      
      vw_i, vb_i, m_t_i, v_t_i, m_b_i, v_b_i=0.0,0.0,0.0,0.0,0.0,0.0
      m_t, v_t, m_hat_w, v_hat_w, m_b,v_b,m_hat_b,v_hat_b = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 

      #below values taken from lecture slides for stability
      eps = 1e-8
      gamma = 0.9
      beta = 0.999
      beta1 = 0.9
      beta2 = 0.999
      train_accuracy, val_accuracy, training_loss ,validation_loss = [] ,[] ,[] ,[]
      
      
      for i in tqdm(range(epochs)):
        for batch in range(0, X_train.shape[1], batch_size):

          batch_images =  X_train[:,batch:batch+batch_size]
          batch_output =  Y_train_encoded[:,batch:batch+batch_size]
          output =  self.forward_propagation(batch_images)
          g_weights,g_biases = self.back_propagation(batch_images,batch_output)
          if optimiser == 'gd':
              self.weights,self.biases=self.stochastic_gradient_descent(self.weights,self.biases,g_weights,g_biases,learning_rate,lambd)
          
          if optimiser == 'mgd':
              self.weights,self.biases,update_w,update_b=self.momentum_based_gradient_descent(self.weights,self.biases,g_weights,g_biases,update_w_i,update_b_i,learning_rate,lambd,gamma)
              update_w_i = update_w
              update_b_i = update_b

          if optimiser == 'ngd':
              w_t,b_t,update_w,update_b=self.nesterov_accelerated_gradient_descent(self.weights,self.biases,g_weights,g_biases,learning_rate,lambd,gamma,update_w_i,update_b_i,batch_images,batch_output)
              self.weights = w_t
              self.biases = b_t
              update_w_i = update_w
              update_b_i = update_b

          if optimiser == 'rmsprop': 
              self.weights,self.biases,vw,vb= self.rmsprop(self.weights,self.biases,g_weights,g_biases,learning_rate,lambd,beta,eps,vw_i,vb_i)
              vw_i=vw
              vb_i=vb

          if optimiser == 'adam':
              self.weights,self.biases,m_t,v_t,m_b,v_b=self.adam(self.weights,self.biases,g_weights,g_biases,beta1,beta2,learning_rate, m_t_i, v_t_i, m_b_i, v_b_i,eps,i,lambd)
              m_t_i=m_t
              v_t_i=v_t
              m_b_i=m_b
              v_b_i=v_b
          
          if optimiser == 'nadam':
              self.weights,self.biases,m_t,v_t,m_b,v_b=self.nadam(self.weights,self.biases,g_weights,g_biases,beta1,beta2,learning_rate, m_t_i,v_t_i, m_b_i,v_b_i, eps, i, lambd)
              m_t_i=m_t
              v_t_i=v_t
              m_b_i=m_b
              v_b_i=v_b                         

        #Calculating accuracies 
        acc1=self.accuracy_score(X_train,y_train)
        train_accuracy.append(acc1)
  
        acc2=self.accuracy_score(X_val,y_val)
        val_accuracy.append(acc2)

        predicted_train = self.forward_propagation(X_train)
        predicted_val = self.forward_propagation(X_val)

        a =self.weights[1:len(self.hidden_layer)]
        b = np.sum([(np.sum((a[i]**2).reshape(1,-1))) for i in range(len(a))])#this is done to update loss function for weight decay problem 
        
        train_loss= loss_function[self.loss_func](Y_train_encoded, y_train, predicted_train, lambd,b )
        val_loss= loss_function[self.loss_func](Y_val_encoded, y_val, predicted_val, lambd,b )

        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        #print(training_loss)
        #print(val_loss)

        print('Epoch {} : training_accuracy = {:.2f}, training_loss = {:.4f},Validation accuracy = {:.2f},Validation loss = {:.4f}'.format(i+1,acc1,train_loss, acc2,val_loss))
        if WandB:
          wandb.log({"val_accuracy": acc2,"accuracy": acc1,"steps":epochs,"train_loss":train_loss,"val_loss":val_loss},)

      
      return train_accuracy,val_accuracy,training_loss, validation_loss


if __name__ == '__main__':    
  
  parser = argparse.ArgumentParser(description='Train a neural network on the MNIST dataset.')
  parser.add_argument('--wandb_entity', type=str, default='shreyashgadgil007', help='Name of the wandb entity')
  parser.add_argument('--wandb_project', type=str, default='CS-6910 A1', help='Name of the wandb project')
  parser.add_argument('--epochs', type=int, default=30, help='No. of epochs for the run')
  parser.add_argument('--batch_size', type=int, default=32, help='No. of batch size for the run')
  parser.add_argument('--learning_rate', type=int, default=0.0001, help='Learning rate to optimize model')
  parser.add_argument('--loss_function', type=str, default='cross_entropy', help='choices:["cross_entropy","square_error"]')
  parser.add_argument('--initialisation', type=str, default='xavier', help='choices:["xavier","random"]')
  parser.add_argument('--optimiser', type=str, default='nadam', help='choices:["gd","mgd","ngd","rmsprop","nadam","adam"]')
  parser.add_argument('--activation', type=str, default='tanh', help='choices:["tanh","sigmoid","relu"]')
  parser.add_argument('--weight_decay', type=int, default=0.0005, help='weight decay value, should be generally low')
  parser.add_argument('--dropout_rate', type=int, default=0, help='dropout value range: (0,1)')
  parser.add_argument('--hidden_layer', type=list, default=[256,256,256], help='No. of hidden layers, format should be in list like:[32,32,32],[64,64,64]') 
  parser.add_argument('--dataset', type=str, default='fashion_mnist', help='choices:["fashion_mnist"],["mnist"]') 
  
  args = parser.parse_args()
  
  X_train,X_test,X_val,Y_train,Y_val,Y_test,Y_train_encoded,Y_val_encoded,Y_test_encoded=dataset_preprocess(args.dataset)
  sweep_config = {
      'method': 'grid', #grid, random,bayes
      'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'  
      },
      'parameters': {
          'epochs': {
              'values': [args.epochs]
          },
          'learning_rate': {
              'values': [args.learning_rate]
          },
          'loss_function':{
              'values':[args.loss_function]
          },
          'initilisation':{
              'values':[args.initialisation]
          },
          'batch_size':{
              'values':[args.batch_size]
          },
          'optimiser': {
              'values': [args.optimiser]
          },
          'activation': {
              'values': [args.activation]
          },
          'hidden_layer': {
              'values': [
                         args.hidden_layer]
          },
          'dropout_rate':{
            'values':[args.dropout_rate]  
                  },
          
          'weight_decay':{
              'values':[args.weight_decay]
          }
          
      }
    }
  def evaluate():
      X_train,X_test,X_val,Y_train,Y_val,Y_test,Y_train_encoded,Y_val_encoded,Y_test_encoded=dataset_preprocess(args.dataset)
      config_defaults = {
          'epochs': 5,
          'input_layer': 784,
          'output_layer': 10,
          'batch_size':64,
          'dropout_rate':0,
          'weight_decay':0.005,
          'learning_rate': 0.0001,
          'hidden_layer':[16,16,16],
          'optimiser':'mgd',
          'activation':'sigmoid',
          'initialisation':'xavier',
          'loss_function':'cross_entropy'
          
       }

      # Initialize a new wandb run
      wandb.init(project=args.wandb_entity, entity=args.wandb_entity,config=config_defaults)
      wandb.run.name = 'Evaluation_run(ED22S016): '+'b_s:'+str(wandb.config.batch_size)+',lr:'+ str(wandb.config.learning_rate)+',ep:'+str(wandb.config.epochs)+'drop:'+str(wandb.config.dropout_rate)+ ',opt:'+str(wandb.config.optimiser)+ ',hl:'+str(wandb.config.hidden_layer)+ ',act:'+str(wandb.config.activation)+',decay:'+str(wandb.config.weight_decay)+',init:'+str(wandb.config.initialisation)+',loss:'+str(wandb.config.loss_function)

      
      # Config is a variable that holds and saves hyperparameters and inputs
      config = wandb.config
      learning_rate = config.learning_rate
      epochs = config.epochs
      hidden_layer = config.hidden_layer
      activation = config.activation
      optimiser = config.optimiser
      input_layer = config.input_layer
      output_layer = config.output_layer
      batch_size = config.batch_size
      weight_decay = config.weight_decay
      loss_function = config.loss_function
      initialisation = config.initilisation
      dropout_rate = config.dropout_rate
       # Model training here
      sweep_network    = NeuralNet(input_layer, hidden_layer, output_layer,initialisation,activation,loss_function,dropout_rate)
      acc1,acc2,train_loss,val_loss  = sweep_network.train(X_train,Y_train,X_val,Y_val,learning_rate,epochs,optimiser,batch_size,weight_decay,WandB=True)
       # print('Acc',acc2)


  sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_entity)
  warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
  wandb.agent(sweep_id, function=evaluate, count=1)
  wandb.finish()  
  

