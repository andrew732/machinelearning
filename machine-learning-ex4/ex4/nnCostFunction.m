function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Get Xt, the raw data plus bias
Xt = [ones(m, 1), X];

%Get a1, the values of hidden layer
result1 = sigmoid(Xt*Theta1');

%Add bias to hidden layers
result1 = [ones(m, 1), result1];

%Activate to get the final results
result = sigmoid(result1*Theta2');

%Initialize error of final layer
delta3 = zeros(num_labels,m);

%Iterate over every test example
for i = 1:m
    
    %Define an array for the actual value, y
    test = [zeros(1,y(i)-1),1,zeros(1,num_labels-y(i))];
    
    %Fill out error of final layer
    delta3(:,i) = (result(i,:)-test)';
    
    %Calcualte cost with formula
    J = J + 1/m * sum(-test'.*log(result(i, :)')-(1-test').*log(1-result(i, :)'));
end
    
%Remove the bias values of theta
temp1 = Theta1(:, 2:end);
temp2 = Theta2(:, 2:end);

%Add in regularization
J = J + lambda/(2*m) * (sum(sum(temp1.*temp1)) + sum(sum(temp2.*temp2)));

%Calculate delta 2 by multiplying Theta2 with delta3
g = Theta2'*delta3;

%Remove bias layer, multiply by derivative of g(z^2)
delta2 = g(2:end,:).*(sigmoidGradient(Xt*Theta1'))';

%Get gradient by dividing by m and multiplying by delta^l+1*a^l
Theta1_grad = 1/m * (delta2*Xt);
Theta2_grad = 1/m * (delta3*result1);

%Regularize gradient 
for j = 2:size(Theta1,2)
    Theta1_grad(:,j) = Theta1_grad(:,j) + lambda/m * Theta1(:,j);
end

for k = 2:size(Theta2,2)
     Theta2_grad(:,k) = Theta2_grad(:,k) + lambda/m * Theta2(:,k);   
end
    














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
