clear all
clc
tic

tau=1;
T=1;

% mu=1.0;
% lambda=1.0;
bin=0.1;
Pmin=0;
Pmax=5;
tmin=-5;

tmax=Pmax - tmin;
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

[tau_2, tau_3, tau_4]=meshgrid([Pmin:bin:Pmax]);
tau_2=reshape(tau_2,[],1);
tau_3=reshape(tau_3,[],1);
tau_4=reshape(tau_4,[],1);

%1st dist
p1=first_dist4m(t2,t3,t4,T);

%2nd dist
f2=load('savedist_4d_old.tsv'); % run the cuda code; make sure to use same bin, mu, lambda, range
toc
%3rd dist
indexy=[];
for k1=1:Pin_point
    for k2=1:Pin_point
        a=[1+Z_point*Z_point*(k1-1)+Z_point*(k2-1):Pin_point+Z_point*Z_point*(k1-1)+Z_point*(k2-1)];
        indexy=[indexy,a]; % indexy give us all input combinations (all {x})
    end
end
indexy=fliplr(indexy);

indexx=[];
    for k=0:(obs_point-1)
        for i=0:(obs_point-1)
            z=[k*Z_point*Z_point+i*Z_point:k*Z_point*Z_point+i*Z_point+(obs_point-1)];
            indexx=[indexx,z];
        end 
    end
toc

%p3=zeros(1,length(tau_2));
parfor i=1:length(tau_2)
	p2=f2(indexx+indexy(i));
	p3(i)=bin*bin*bin*sum(p1.*p2');
end

toc

%4th dist
p4=fourth_dist4m(tau_2,tau_3,tau_4,tau);
bin*bin*bin*sum(p4)

%5th dist
p=bin*bin*bin*sum(p4.*p3)



