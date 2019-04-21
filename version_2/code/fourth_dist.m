function ps=fourth_dist(a,b,tau)
    if ~exist('tau')
        tau = 1;
    end
    for i =1:size(a,1)
    if (b(i)>=a(i))
        ps(i)=(1/tau.^2)*(exp(-1*b(i)/tau));
    else
        ps(i)=0;
    end
    end
end