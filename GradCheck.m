function [both, diff] = GradCheck(X, y, theta)

  %Numerical gradient...
  J = @(p)CostFunc(X, y, p);
  numgrad = zeros(size(theta));
  perturb = zeros(size(theta));
  e = 1e-4;
  for p = 1:numel(theta)
      % Set perturbation vector
      perturb(p) = e;
      [loss1, ~] = J(theta - perturb);
      [loss2, ~] = J(theta + perturb);
      % Compute Numerical Gradient
      numgrad(p) = (loss2 - loss1) / (2*e);
      perturb(p) = 0;
  end

  %Graient from backpropagation...
  [~, bg] = CostFunc(X, y,theta);

  both = [bg numgrad];
  diff = norm(numgrad-bg)/norm(numgrad+bg);
end
