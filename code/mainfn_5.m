% main function
% longer observation and input time range

clear all
clc
tic

tau=1;
T=1;

% mu=1.0;
% lambda=1.0;
bin=0.10;
Pmin=0;
Pmax=20;
tmin=-10;
tmax=30;

Pin_point=(Pmax-Pmin)/bin +1;
obs_point=(tmax-tmin)/bin+1;
Zmax=tmax;
Zmin=tmin-Pmax;
Z_point=(Zmax-Zmin)/bin+1;


% parameters
[t2,t3]=meshgrid([tmin:bin:tmax]);
t2=reshape(t2,[],1);
t3=reshape(t3,[],1);

[tau_2,tau_3]=meshgrid([Pmin:bin:Pmax]);
tau_2=reshape(tau_2,[],1);
tau_3=reshape(tau_3,[],1);

%1st dist
p1=first_dist(t2,t3,T);

%2nd dist
f2=load('savedist_3d.tsv'); % run the cuda code; make sure to use same bin, mu, lambda, range

%3rd dist
indexy=zeros(1,Pin_point*Pin_point);
for k=1:Pin_point
a=[1+Z_point*(k-1):Pin_point+Z_point*(k-1)];
indexy((k-1)*Pin_point+1:k*Pin_point)= a; % indexy give us all input combinations (all {x})
end
indexy=fliplr(indexy);

indexx=zeros(1,obs_point*obs_point);
    for i=0:(obs_point-1)
        z=[i*Z_point:i*Z_point+(obs_point-1)];
        indexx(i*obs_point+1:(i+1)*obs_point) = z;
    end
    
toc
p3=zeros(1,length(tau_2));
for i=1:length(tau_2)
	p2=f2(indexx+indexy(i));
	p3(i)=bin*bin*sum(p1.*p2');
end

toc
%4th dist
p4=fourth_dist(tau_2,tau_3,tau);
bin*bin*sum(p4)

%5th dist
p=bin*bin*sum(p4.*p3)
toc



