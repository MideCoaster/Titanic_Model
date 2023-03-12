function [Ftheta, jhist] = GradintDesc(X, y , theta, alpha, inum)
jhist = [];
for i =1:inum
    [j, grad] = CostFunc(X, y,theta);
    theta = theta - alpha*grad;
    jhist = [jhist; j];
    if (mod(i,1000)==0)
      fprintf("Iteration     %d | cost: %d\n", i, j);
    end
end
Ftheta = theta;
end
