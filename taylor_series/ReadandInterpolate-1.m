function NewSaccade=ReadandInterpolate(saccadeInput)
    NewSaccade=[];
    saccadeRecord=[];

    saccadeRecord.x=saccadeInput.saccade(:,1);
    saccadeRecord.y=saccadeInput.saccade(:,2);
    saccadeRecord.t=saccadeInput.elapsedtime;

    len = length(saccadeRecord.x);
    saccadeRecord.t = saccadeRecord.t- saccadeRecord.t(1);

    % interpolate to make up missing samples
    ct=1;
    for n=1:len
       NewSaccade.x(ct) = saccadeRecord.x(n);
       NewSaccade.y(ct) = saccadeRecord.y(n);
       NewSaccade.t(ct) = saccadeRecord.t(n);
       ct = ct+1;
       if n<len && saccadeRecord.t(n+1) - saccadeRecord.t(n)==2
           NewSaccade.x(ct) = (saccadeRecord.x(n) + saccadeRecord.x(n+1))/2;
           NewSaccade.y(ct) = (saccadeRecord.y(n) + saccadeRecord.y(n+1))/2;
           NewSaccade.t(ct) = (saccadeRecord.t(n) + saccadeRecord.t(n+1))/2;
           ct = ct+1;
       elseif n<len && saccadeRecord.t(n+1) - saccadeRecord.t(n)>2
           disp('Miss more than one sample');
           break;
       end
    end
    
end