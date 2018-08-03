% main function
% longer observation and input time range

clear all
clc
tic

tau=0.4;
T=0.6;

% mu=1.0;
% lambda=1.0;
bin=0.05;
Pmin=0;
Pmax=6;
tmin=-4;
tmax=10;
Pin_point=(Pmax-Pmin)/bin +1;
Z_point=401;
obs_point=(tmax-tmin)/bin+1;

% parameters
[t2,t3]=meshgrid([tmin:0.05:tmax]);
t2=reshape(t2,[],1);
t3=reshape(t3,[],1);

[tau_2,tau_3]=meshgrid([Pmin:0.05:Pmax]);
tau_2=reshape(tau_2,[],1);
tau_3=reshape(tau_3,[],1);

%1st dist
p1=first_dist(t2,t3,T);

%2nd dist
% f2=load('savedist_3d_4_05.tsv');
f2=load('savedist_3d.tsv');
%f2=Q = second_dist(mu, lambda, t2, t3, 0, 0 )

%3rd dist

indexy=[];
for k=1:Pin_point
a=[1+Z_point*(k-1):Pin_point+Z_point*(k-1)];
indexy=[indexy,a]; % indexy give us all input combinations (all {x})
end
indexy=fliplr(indexy);

indexx=[];
    for i=0:(obs_point-1)
        z=[i*Z_point:i*Z_point+(obs_point-1)];
        indexx=[indexx,z];
    end 

for i=1:length(tau_2)
	p2=f2(indexx+indexy(i));
	p3(i)=bin*bin*sum(p1.*p2');
end
	
%4th dist
p4=fourth_dist(tau_2,tau_3,tau);
bin*bin*sum(p4)

%5th dist
p=bin*bin*sum(p4.*p3)
toc



