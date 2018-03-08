% regression
function p_agc=PredictAngleC(agc,step)
    pn=5; 
    len=length(agc);
    r=0.5^step;    
    
    if len <=pn
        p_agc=agc(len)+(agc(len)-agc(max(1,len-1)))*r;
    else
        b=[1.5; -1.0; 1.2; -0.8; 0.1];%pn=5; 
        observation=zeros(pn,1);
        for n=1:pn
            % mean+delta
            for i=0:n-1
                observation(n)=observation(n)+agc(len-i);
            end
            observation(n)=observation(n)+(agc(len)-agc(len-n))*r;
            observation(n)=observation(n)/n;
        end
        p_agc=sum(observation.*b);
    end
end
