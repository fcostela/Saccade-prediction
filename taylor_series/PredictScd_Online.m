%   input:
%       KnownSaccade - the coordinates of previous saccade positions
%       KnownSaccade should include at least one point, and this function can work well when the point number is larger than 2
%       DelayTime - the delay time of saccade prediction, the default value is 10
%   output:
%       PreSaccade - the position of the predicted saccade position

function PreSaccade=PredictScd_Online(KnownSaccade, DelayTime)
        pn=5; 
        % the ideal numner of samples is pn+3
        ideal_num=pn+3;
        len = length(KnownSaccade.x); 
        
        if len>ideal_num
            KnownSaccade.x=KnownSaccade.x(len-ideal_num+1:len);
            KnownSaccade.y=KnownSaccade.y(len-ideal_num+1:len);
            len=ideal_num;
        end
        
%         if len==1
%             KnownSaccade.x=repmat(KnownSaccade.x(1), len,1);
%         elseif len<pn
%             x_add=zeros(1,in-len);
%             y_add=zeros(1,in-len);
%             x_diff=(KnownSaccade.x(end)-KnownSaccade.x(1))/(len-1);
%             y_diff=(KnownSaccade.y(end)-KnownSaccade.y(1))/(len-1);
%             for idx=(in-len):-1:1
%                 x_add(idx)=KnownSaccade.x(1)-x_diff*(idx-in+len+1);
%                 y_add(idx)=KnownSaccade.y(1)-y_diff*(idx-in+len+1);
%             end
%             
%             KnownSaccade.x=[x_add, KnownSaccade.x(len-in+1:len)];
%             KnownSaccade.y=[y_add, KnownSaccade.y(len-in+1:len)];
%         else
%             KnownSaccade.x=KnownSaccade.x(len-in+1:len);
%             KnownSaccade.y=KnownSaccade.y(len-in+1:len);
%         end


        if len>1
            dx=KnownSaccade.x(2:end)-KnownSaccade.x(1:end-1);
            dy=KnownSaccade.y(2:end)-KnownSaccade.y(1:end-1);

            %velocity from t(2) to t(len) 
            vlc=sqrt(dx.^2+dy.^2); 

            % rotation anlge from t(2) to t(len) 
            angle=atan(dy./(dx+~dx))/pi*180;
            angle=angle+ (dx<0)*180;
            % align the rotation angle
            for i=2:len-1
                if angle(i)-angle(i-1)>90
                    angle(i)=angle(i)-360;
                elseif angle(i)-angle(i-1)<-90
                    angle(i)=angle(i)+360;
                end
            end
        else
            dx=0;
            dy=0;
        end
        
        if len<=2
            PreSaccade.x=KnownSaccade.x(end)+dx(end)*DelayTime;
            PreSaccade.y=KnownSaccade.y(end)+dy(end)*DelayTime;
            return;
        end

        acc=vlc(2:end)-vlc(1:end-1);            % velocity acceleration
        agcc=angle(2:end)-angle(1:end-1); % angle acceleration

        pdl_vlc=zeros(1,DelayTime);     % predict list of velocity
        pdl_agl=zeros(1,DelayTime);         % predict list of  angle rotation
        
        alist=acc;
        aglist=agcc;
        for delta=1:DelayTime            
            a=PredictAcceleration(alist);
            if delta==1
                pdl_vlc(delta)=vlc(end)+a; % velocity in a unit time
            else
                pdl_vlc(delta)=pdl_vlc(delta-1)+a; % velocity in a unit time
            end

            alist=[alist, a];

            ag =PredictAngleC(aglist,delta);
            if delta==1
                pdl_agl(delta)=angle(end)+ag;
            else
                pdl_agl(delta)=pdl_agl(delta-1)+ag;
            end
            aglist=[aglist, ag];
        end
        [PreSaccade.x, PreSaccade.y]=Compose(KnownSaccade.x(end), KnownSaccade.y(end), pdl_vlc, pdl_agl);
end