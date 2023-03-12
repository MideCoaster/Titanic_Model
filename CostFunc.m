function [J, grad] = CostFunc (x, y, theta)
  [m,n] = size(x);
  lambda = 0;
  a = n*100;

  t1 = reshape(theta(1:a),100,n);
  t2 = reshape(theta(a+1:a+10100),100,101);
  t3 = reshape(theta(a+10101:end),1,101);

  a1 = x; %m*n
  a2 = [ones(size(y)) sigmoid(a1*t1')]; %m*26
  a3 = [ones(size(y)) sigmoid(a2*t2')]; %m*26
  a4 = sigmoid(a3*t3'); %m*1

  J = (-1/m)*sum(y.*log(a4) + (1-y).*log(1-a4)) + (lambda/(2*m))*sum(theta.^2); %scalar

  d4 = a4 - y; %m*1
  d3 = d4*t3.*(a3.*(1-a3)); d3= d3(:,2:end); %m*25
  d2 = d3*t2.*(a2.*(1-a2)); d2 = d2(:,2:end); %m*25

  D1 = (1/m)* d2'*a1; D1(:, 2:end) = D1(:, 2:end) + (lambda/m)*t1(:,2:end);
  D2 = (1/m)* d3'*a2; D2(:, 2:end) = D2(:, 2:end) + (lambda/m)*t2(:,2:end);
  D3 = (1/m)* d4'*a3; D3(:, 2:end) = D3(:, 2:end) + (lambda/m)*t3(:,2:end);

  grad = [D1(:); D2(:); D3(:)];
end
