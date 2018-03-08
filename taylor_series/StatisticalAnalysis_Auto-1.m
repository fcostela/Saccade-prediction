% parameters
curvature_low=0.1;
curvature_high=0.2;
vlcerr_lmt=0.004; %vlc error limit
pn_lmt=25; %point number limit, used to select samples

amplitude_lmt=40;
sec_n=10; %section number, used to divide the amplitude
ErrorSum=0;

avn=0;

idx=54090;
Error=[];
while idx<=200740   %   the index of items
    curvature=saccadeIndex(idx).curvature;
    vlcerr=saccadeIndex(idx).velocity.err;

    if length(saccadeIndex(idx).saccade)<pn_lmt || ( curvature<curvature_high  || vlcerr>=vlcerr_lmt)
        idx=idx+1;
        continue;
    end
    
    % read and interpolate to make up missing samples
    NewSaccade=ReadandInterpolate(saccadeIndex(idx));
    amplitude=getAmplitude(NewSaccade);
    if amplitude>amplitude_lmt %|| any(idx==special)
        idx=idx+1;
        continue;
    end
    
   DelayTime=10;
   len = length(NewSaccade.x); 
   PreSaccade.x=NewSaccade.x;
   PreSaccade.y=NewSaccade.y;
   for i=1:len-DelayTime
       KnownSaccade.x=NewSaccade.x(1:i);
       KnownSaccade.y=NewSaccade.y(1:i);
%        saccade=PredictScd(KnownSaccade, DelayTime);
       saccade=PredictScd_Online(KnownSaccade, DelayTime)
       PreSaccade.x(i+DelayTime)=saccade.x;
       PreSaccade.y(i+DelayTime)=saccade.y;
   end
   
   figure, plot(NewSaccade.x,NewSaccade.y,'or',PreSaccade.x(DelayTime+1:end),PreSaccade.y(DelayTime+1:end),'*b');
   
    idx=idx+1
end


