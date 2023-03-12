function retval = PPredict (x, theta, thresh)

  [m, n] = size(x);
  a = n*100;

  t1 = reshape(theta(1:a),100,n);
  t2 = reshape(theta(a+1:a+10100),100,101);
  t3 = reshape(theta(a+10101:end),1,101);

  a1 = x; %m*n
  a2 = [ones(m,1) sigmoid(a1*t1')]; %m*26
  a3 = [ones(m,1) sigmoid(a2*t2')]; %m*26
  a4 = sigmoid(a3*t3'); %m*1

  retval = (a4>thresh);
end
