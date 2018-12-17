function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp = zeros(2,1);
% update=zeros(2,1);
for iter = 1:num_iters   %num_iters�ε���

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    fprintf(' �� %d �� iteration   ',iter);
    
        temp(1,1) = theta(1,1) - alpha/m*sum(X*theta - y);
        temp(2,1) = theta(2,1) - alpha/m*sum((X*theta-y) .* X(:,2));
        theta = temp;
   
%    ������ѭ��ʵ���ǲ��Եģ���Ϊÿһ��iֵ����Ӧ�Ⱥ��ұߵ�updateֵ���ǲ�ͬ�ģ���
%    ���ڵ����Ĺ�ʽ  theta(0) = theta(0) - alpha/m*��(i=1:m)(X*theta - y)��˵��
%    �ڼ����1��m�����ʱ��theta(0)��û�иı�ģ����Բ������������ʵ��
%      for i=1:m
% 
% 		update=update+(X(i,:)*theta-y(i))*X(i,:)';    
% 
%     end
% 
%     update=update*alpha/(m);
% 
%     theta=theta-update; 
%     
%    


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
   

end

end
