
function a= first_dist4m(t2,t3,t4,T)
    i=1;
    a= zeros(1,length(t2));
    parfor i=1:length(t2)
            X= sort([t2(i), t3(i), t4(i), 0]);
            t1s = X(1); t2s = X(2); t3s = X(3); t4s = X(4);
            a(i) = 1- myheaviside (t4s - t2s - T);
    end
end
