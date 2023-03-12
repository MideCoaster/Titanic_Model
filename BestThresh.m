function [bestEpsilon ,bestF1] = BestThresh(yval, pval)

  bestEpsilon = 0;
  bestF1 = 0;
  F1 = 0;

  stepsize = (max(pval) - min(pval)) / 1000;

  % F1-SCORE = 2Tp/(2Tp + (Fp+Fn)) = 2PR/P+R. So...
  % The Above Equation is my derived simplification of the F1-Score equation.
  
  for epsilon = min(pval):stepsize:max(pval)
      p = (pval>epsilon);
      fpn= (sum(yval~=p)); % Fp + Fn
      tp = 0;

      for i = 1:length(p)
        if (p(i)==1)
          if (yval(i)==1)
            tp = tp+1;
          end
        end
      end

      F1 = (2*tp)/(2*tp + fpn);

      if (F1 > bestF1)
         bestF1 = F1;
         bestEpsilon = epsilon;
      end
  end

end
