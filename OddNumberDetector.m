clear; close all; clc;
load("Titanic.mat");

deg = 20;
Xtrain = polyfeature (Xtrain, deg);
X = polyfeature (X, deg);
Xtest = polyfeature (Xtest, deg);
a = [X; Xtest; Xtrain];

miin = mean(a,1);
[~, N] = size(Xtrain);
v = ((N-1)/N).*var(a,1);

Xtrain = (Xtrain-miin)./v;
Xtrain = [ones(size(ytrain)) Xtrain];
[m, n] = size(Xtrain);
Xtest = (Xtest-miin)./v;
Xtest = [ones(size(ytest)) Xtest];

itheta = rand(n*100 + 10201, 1);

%fprintf("Implementing Gradient Checking...\n\n");
%[both, diff] = GradCheck(Xtrain, ytrain, itheta);
%disp(both(1:100,:));
%fprintf("The difference is %d\n\n", diff);
%fprintf("If the gradients are correct, the diff should be very small!\n");
%fprintf("paused! Press enter to continue\n\n");


options = optimset("GradObj", "on", "MaxIter", 1000);
[theta, Hist] = fmincg(@(t)(CostFunc(Xtrain, ytrain, t)), itheta, options);


%alpha = 0.3; inum = 20000;
%[theta, Hist] = GradintDesc(Xtrain, ytrain , itheta, alpha, inum);

fprintf("Paused! press enter to continue\n\n");
pause;

plot((1:length(Hist))', Hist, "b-");

fprintf("Paused! press enter to continue\n\n");
pause;

threshold = 0.5;
[hypo, p, acc] = predict(Xtest,ytest,theta, threshold);
truth = [hypo p ytest];
disp(truth);

fprintf("The accuracy is %d percent\n\n", acc);
fprintf("Paused! press enter to continue\n\n");
pause;

fprintf("Searching for the best Threshold\n\n");
[bestthresh ,bestF1] = BestThresh(ytest, p);
disp([hypo ytest]);

fprintf("The best threshold = %d, and the F1-score = %d\n\n", bestthresh, bestF1)
[hypo, p, acc] = predict(Xtest,ytest,theta, bestthresh);
fprintf("And The New Accuracy is %d percent\n\n", acc);

if acc>=80
  X = (X-miin)./v;
  X = [ones(size(X,1),1) X];
  hypoth = PPredict(X, theta, bestthresh);
  solute = [(892:1309)' hypoth];
  save "Solution1.csv" solute

  hypoth = PPredict(X, theta, threshold);
  solute = [(892:1309)' hypoth];
  save "Solution2.csv" solute

  save "Trainedweights.mat" theta bestthresh
end


