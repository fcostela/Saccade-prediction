% ty to find the regulation
function a=PredictAcceleration(acc)  
len=length(acc); 
    acc_lmt=0.05; % acc lmt 
 %      0.68; 0.80;1.15;1.19
    if acc(len)>=acc(max(1,len-1))&& acc(max(1,len-1))>=0  % away from 0
        change_decrease=0.8591;
        a=acc(len)+(acc(len)-acc(max(1,len-1)))*change_decrease-0.0005;
    elseif acc(len)<=acc(max(1,len-1)) && acc(max(1,len-1))<=0 % away from 0
        change_decrease=0.7657;    
        a=acc(len)+(acc(len)-acc(max(1,len-1)))*change_decrease+0.0003;
    elseif  acc(len)<=acc(max(1,len-1))&& acc(len)>=0% approach to 0
        chang_enlarge=0.8835;
        a=acc(len)+(acc(len)-acc(max(1,len-1)))*chang_enlarge-0.0008;
    elseif acc(len)>=acc(max(1,len-1))&& acc(len)<=0 % approach to 0, not linear!!!
        chang_enlarge=1.0121;
        a=acc(len)+(acc(len)-acc(max(1,len-1)))*chang_enlarge-0.0003;
    else
        a=acc(len)+(acc(len)-acc(max(1,len-1)));
    end
    a=sign(a)*min(abs(a), acc_lmt);
end