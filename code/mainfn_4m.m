clear all
clc
tic

tau=1;
T=1;

% mu=1.0;
% lambda=1.0;
bin=0.1;
Pmin=0;
Pmax=15;
tmin=-10;
tmax=25;

Pin_point=(Pmax-Pmin)/bin +1;
obs_point=(tmax-tmin)/bin+1;
Zmax=tmax;
Zmin=tmin-Pmax;
Z_point=(Zmax-Zmin)/bin+1;


% parameters
[t2,t3,t4]=meshgrid([tmin:bin:tmax]);
t2=reshape(t2,[],1);
t3=reshape(t3,[],1);
t4=reshape(t4,[],1);

[tau_2,tau_3, tau_4]=meshgrid([Pmin:bin:Pmax]);
tau_2=reshape(tau_2,[],1);
tau_3=reshape(tau_3,[],1);
tau_4=reshape(tau_4,[],1);

%1st dist
p1=first_dist4m(t2,t3,t4,T);

%2nd dist
f2=load('savedist_4d.tsv'); % run the cuda code; make sure to use same bin, mu, lambda, range

%3rd dist

%4th dist
p4=fourth_dist4m(tau_2,tau_3,tau_4,tau);
bin*bin*sum(p4)



