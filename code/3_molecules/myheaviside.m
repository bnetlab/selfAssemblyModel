function out = myheaviside(in)
  out = heaviside(in);
  out(~in) = 0;
end