function [prediction, p, accuracy] = predict (x, y, theta, thresh)
  [m, n] = size(x);
  a = n*100;

  t1 = reshape(theta(1:a),100,n);
  t2 = reshape(theta(a+1:a+10100),100,101);
  t3 = reshape(theta(a+10101:end),1,101);

  a1 = x; %m*n
  a2 = [ones(size(y)) sigmoid(a1*t1')]; %m*26
  a3 = [ones(size(y)) sigmoid(a2*t2')]; %m*26
  a4 = sigmoid(a3*t3'); %m*1

  prediction = (a4>thresh);
  accuracy = ((sum(prediction==y))/m)*100;
  p = a4;
end
