% calculate equation 2


function Q = second_dist(mu, lambda, t2, t3, tau2, tau3 )
for i =1:length(t2)
    Q(i)=integral(@(x)pdf('InverseGaussian',x,mu,lambda).*pdf('InverseGaussian',x+t2(i)-tau2,mu,lambda).*pdf('InverseGaussian',x+t3(i)-tau3,mu,lambda),0,50);
end
end
