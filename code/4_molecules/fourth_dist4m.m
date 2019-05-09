function ps=fourth_dist4m(a,b,c,tau)
    if ~exist('tau')
        tau = 1;
    end
    for i =1:size(a,1)
    if (b(i)>=a(i) && c(i)>=a(i) && c(i)>=b(i))
        ps(i)=(1/tau.^3)*(exp(-1*c(i)/tau));
    else
        ps(i)=0;
    end
    end
end