% main function
% longer observation and input time range

clear all;clc;
tic

tau1=[0.5:0.25:5];
T1=[0.5:0.25:5];
[tauX,TX] = meshgrid(tau1,T1);
tauX=reshape(tauX,numel(tauX),[]);
TX=reshape(TX, numel(TX),[]);

parfor loop=1:numel(tauX)
tau=tauX(loop)
T=TX(loop)

% integration range
bin=0.05;
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
[t2,t3]=meshgrid([tmin:0.05:tmax]);
t2=reshape(t2,[],1);
t3=reshape(t3,[],1);

[tau_2,tau_3]=meshgrid([Pmin:0.05:Pmax]);
tau_2=reshape(tau_2,[],1);
tau_3=reshape(tau_3,[],1);

%1st dist
p1=first_dist(t2,t3,T);

%2nd dist
f2=load('savedist_3d.tsv');

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

p3=zeros(1,length(tau_2))
for i=1:length(tau_2)
	p2=f2(indexx+indexy(i));
	p3(i)=bin*bin*sum(p1.*p2');
end
	
%4th dist
p4=fourth_dist(tau_2,tau_3,tau);
bin*bin*sum(p4)

%5th dist
    p(loop)=0.05*0.05*sum(p4.*p3)
end
toc
clearvars -except p tauX TX



