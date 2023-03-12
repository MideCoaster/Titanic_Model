function nx = polyfeature (x, deg)
  [m, n]  =  size(x);
  a = [];
  if (deg==1)
    nx = x;
  else
    for k = 1:deg
      c = x.^(k-1);
      for  i = 1:n
        d = x(:,i).*c(:,i:end);
        a = [a d];
      end
    end
    nx = a;
  end
end
