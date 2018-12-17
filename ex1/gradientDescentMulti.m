function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);  %记录每一次迭代对应的代价函数值
n = size(X,2);  %特征的数目，其中第一个特征的值全为1，是我们为计算方便而加上去的
temp = zeros(size(theta,2),1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

        
%     for i = 1:n
%         temp(i,1) = theta(i,1) - alpha/m*sum((X*theta - y).*X(:,i)); %要注意sum()包含的范围,最后乘的X(:,i)是X的列向量，对应于每一列参数theta,而且是点乘
%     end
%     theta = temp;
        theta = theta - X'*(X*theta-y)/m*alpha;

    








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta); %每一次迭代记录下对应的代价函数

end

end
