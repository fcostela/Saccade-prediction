%   input:
%       KnownSaccade - the coordinates of previous saccade positions
%       KnownSaccade should include at least one point, and this function can work well when the point number is larger than 2
%       DelayTime - the delay time of saccade prediction, the default value is 10
%   output:
%       PreSaccade - the position of the predicted saccade position

function PreSaccade=PredictScd_Peng(KnownSaccade, DelayTime)
    len = length(KnownSaccade.x); 
    time=0:len-1;
    
    x = KnownSaccade.x;
    y = KnownSaccade.y;
    SacData = complex(x, y);

    if length(time)<6
        x = KnownSaccade.x(end);
        y = KnownSaccade.y(end);
        PreSaccade.x=x;
        PreSaccade.y=y;
        return
    end

    NormData = SacData-SacData(1);
    % Compute the distange
    FitData = abs(NormData);
    [S, cnt, xf] = MyLMFsolve(time, FitData);
    % xf = lsqnonlin(errFun,startingVals); % build-in function in matlab

    % Compute the angle, based only on a line between the first and last points
    ang = angle(NormData(end));

    % Predict the distance
    dis = saccadefun(time(end)+DelayTime, xf);

    % Compute the position it's supposed to be in.
    p = dis*cos(ang)+1i*dis*sin(ang)+SacData(1);

    PreSaccade.x=real(p);
    PreSaccade.y=imag(p);
end

function y = saccadefun(t, xf)

y = abs(xf(1).*(1-exp(-((t./xf(2)).^xf(3)))));
end


function [S,cnt,xf] = MyLMFsolve(time,FitData)
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% % x=[x0 15 2]';
% % r  = x0.*(1-exp(-(time./15).^2)) - FitData;           % Residuals at starting point

%Begining of the modification-------------------------------------------
x0 = FitData(end);
x1 = time(end);
x=[x0 x1 1.5]';
r  = x0.*(1-exp(-(time./x1).^1.5)) - FitData;           % Residuals at starting point
%End of modification --------------------------------------------------


r = r';
%~~~~~~~~~~~~~~~~~
S  = r'*r;                          %the data needed in DegreePre function
%~~~~~~~~~~~~~~~~~
%%%%%%%%generate A D and v by Jacobi
%~~~~~~~~~~~~~~~~~~Jacobi begins
epsx=(1e-4)*ones(3,1);
lx=length(x);
J=zeros(length(r),lx);
for k=1:lx
    dx=.25*epsx(k);
    xd=x;
    xd(k)=xd(k)+dx;
    rd=xd(1).*(1-exp(-(time./xd(2)).^xd(3))) - FitData;
    rd = rd';
    
    %   ~~~~~~~~~~~~~~~~
    J(:,k)=((rd-r)/dx);
end
%~~~~~~~~~~~~~~~~~~~~Jacobi ends
A = J.'*J;                    % System matrix
v = J.'*r;

D = diag(diag(A));        % automatic scaling
for i = 1:lx
    if D(i,i)==0, D(i,i)=1; end
end
epsf = 1e-7;
Rlo = 0.25;
Rhi = 0.75;
l=1;      lc=.75;
cnt = 0;

d = (1e-4)*[1 0 0]';     %in this step,we just need d=1e-4,but to match
%3*1 size in line 43,we let d = (1e-4)*[1 0 0]
maxit = 10;
nfJ = 2;
while cnt<maxit && ...          %   MAIN ITERATION CYCLE
        any(abs(d) >= epsx) && ...      %%%%%%%%%%%%%%%%%%%%
        any(abs(r) >= epsf)
    d  = (A+l*D)\v;             %   negative solution increment
    xd = x-d;
    rd = xd(1).*(1-exp(-(time./xd(2)).^xd(3))) - FitData;
    rd = rd';
    %   ~~~~~~~~~~~~~~~~~~~
    nfJ = nfJ+1;
    Sd = rd.'*rd;
    dS = d.'*(2*v-A*d);         %   predicted reduction
    
    R  = (S-Sd)/dS;
    if R>Rhi                    %   halve lambda if R too high
        l = l/2;
        if l<lc, l=0; end
    elseif R<Rlo                %   find new nu if R too low
        nu = (Sd-S)/(d.'*v)+2;
        if nu<2
            nu = 2;
        elseif nu>10
            nu = 10;
        end
        if l==0
            lc = 1/max(abs(diag(inv(A))));
            l  = lc;
            nu = nu/2;
        end
        l = nu*l;
    end
    %~~~~~~~~~~~~~~~~
    cnt = cnt+1;    %the data needed in DegreePre function
    %~~~~~~~~~~~~~~~~
    if Sd<S
        S = Sd;
        x = xd;
        r = rd;
        %~~~~~~~~~~~~~~~~~~~~~~~Jacobi begins
        lx=length(x);
        J=zeros(length(r),lx);
        for k=1:lx
            dx=.25*epsx(k);
            xd=x;
            xd(k)=xd(k)+dx;
            rd=xd(1).*(1-exp(-(time./xd(2)).^xd(3))) - FitData;
            rd = rd';
            
            J(:,k)=((rd-r)/dx);
        end
        %~~~~~~~~~~~~~~~~~~~~~~~Jacobi ends
        
        nfJ = nfJ+1;
        A = J'*J;
        v = J'*r;
    end
end %   while

xf = x;                         %   final solution
end





% 
% %   input:
% %       KnownSaccade - the coordinates of previous saccade positions
% %       KnownSaccade should include at least one point, and this function can work well when the point number is larger than 2
% %       DelayTime - the delay time of saccade prediction, the default value is 10
% %   output:
% %       PreSaccade - the position of the predicted saccade position
% 
% function PreSaccade=PredictScd_Peng(KnownSaccade, DelayTime)
%     pixelsPerCm = 35.1; % ASUS monitor
%     x = cmToVisualAngle(KnownSaccade.x / pixelsPerCm, 60);
%     y = cmToVisualAngle(KnownSaccade.y / pixelsPerCm, 60);
%     len = length(KnownSaccade.x); 
%     time=0:len-1;
%     
%     SacData = complex(x, y);
% 
%     if length(time)<6
%         x = KnownSaccade.x(end);
%         y = KnownSaccade.y(end);
%         PreSaccade.x=x;
%         PreSaccade.y=y;
%         return
%     end
% 
%     NormData = SacData-SacData(1);
%     % Compute the distange
%     FitData = abs(NormData);
%     [S, cnt, xf] = MyLMFsolve(time, FitData);
%     % xf = lsqnonlin(errFun,startingVals); % build-in function in matlab
% 
%     % Compute the angle, based only on a line between the first and last points
%     ang = angle(NormData(end));
% 
%     % Predict the distance
%     dis = saccadefun(time(end)+DelayTime, xf);
% 
%     % Compute the position it's supposed to be in.
%     p = dis*cos(ang)+1i*dis*sin(ang)+SacData(1);
% 
%     x = visualAngleToCm(real(p), 60) * pixelsPerCm;
%     y = visualAngleToCm(imag(p), 60) * pixelsPerCm;
%     PreSaccade.x=x;
%     PreSaccade.y=y;
% end
% 
% function y = saccadefun(t, xf)
% 
% y = abs(xf(1).*(1-exp(-((t./xf(2)).^xf(3)))));
% end
% 
% %Compressed exponential function
% function F = Cexp(c,xdata)
% F = c(1).*(1-exp(-(xdata./c(2)).^c(3)));
% end
% 
% function size = visualAngleToCm(angle,distanceFromMonitor)
%     % Take a visual angle, convert into a size on the monitor in cm based on
%     % distanceFromMonitor. checkPixelSize can be useful for converting between
%     % cm and pixels.
%     %
%     % Usage: size = visualAngleToCm(angle,distanceFromMonitor)
% 
%     if nargin < 2
%         distanceFromMonitor = 65;
%     end
%     % size is in cm
% 
%     angle = angle / (180/pi);
%     angle = angle / 2; 
%     size = 2 * (tan(angle) * distanceFromMonitor);
% end
% 
% 
% function angle = cmToVisualAngle(size,distanceFromMonitor)
%     % Take a size on the monitor in cm, convert to a visual angle using
%     % distanceFromMonitor. checkPixelSize can be useful for converting between
%     % cm and pixels.
%     %
%     % Usage: angle = cmToVisualAngle(size,distanceFromMonitor)
% 
%     if nargin < 2
%         distanceFromMonitor = 65;
%     end
%     % size is in cm
% 
%     angle = atan((size/2) / distanceFromMonitor);
%     angle = angle * 2;
%     angle = angle * (180 / pi); % convert from radians to degrees
% end

















    




    
