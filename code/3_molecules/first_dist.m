% calculate equation 1

function a= first_dist(t2,t3,T)
    i=1;
    while (i<= length(t2))
        if (t2(i) >=0  && t3(i) >=0) || (t2(i) <0 && t3(i) < 0)
            a(i)=1- myheaviside (abs (t3(i) -t2(i))- T);
        end
        if (t2(i) <0 && t3(i) >=0)
            a(i)=1- myheaviside (t3(i)- T);
        end
        if (t2(i) >= 0 && t3(i) < 0)
            a(i)= 1- myheaviside (t2(i) - T);
        end
        i=i+1; 
    end
end
