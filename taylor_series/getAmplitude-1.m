function amplitude=getAmplitude(NewSaccade)
        dx=NewSaccade.x(2:end)-NewSaccade.x(1:end-1);
        dy=NewSaccade.y(2:end)-NewSaccade.y(1:end-1);
        vlc=sqrt(dx.^2+dy.^2);
        amplitude=sum(vlc);
end