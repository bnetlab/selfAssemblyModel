% main function
% calculate dist 5
% for better precision  please decrease bin size and increse range in main
% and third dist function

mu=0.5;
lambda=0.5;
T=1;

[tau_2,tau_3]=meshgrid([0:0.1:1]);
tau_2=reshape(tau_2,[],1);
tau_3=reshape(tau_3,[],1);

ps=fourth_dist(tau_2,tau_3);

for i=1:length(tau_2)
    p(i)=third_dist(mu, lambda, tau_2(i), tau_3(i));
    fprintf('.')
end

prob=0.1*0.1* sum(ps.*p)
