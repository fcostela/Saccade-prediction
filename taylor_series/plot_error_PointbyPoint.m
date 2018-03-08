% This script plots the error distribution between predicted position and the landing position of the saccades

% load saccadeIndex5

%cols.mat has the error with new algorithm 

load '/Users/FranciscoCostela/Desktop/GazeContigentDisplay/SaccadeAnalysis/saccadeDatabase_without_odd.mat';
saccade = [];
elapsedtime = [];

warning off
%saccadeIndex = all;
for i=1:length(sel_saccades)%size(saccadeIndex,2)
    saccade(i).x = sel_saccades(i).saccade(:,1);
    saccade(i).y = sel_saccades(i).saccade(:,2);
    %elapsedtime(i) = length(saccadeIndex(i).elapsedtime);
end

j = 1;
difference = [];

disp('Calling Shuhang');

% Calling Shuhang algorithm

% for i=1:length(sel_saccades)%size(sel_saccades,2)
%  
%         if ~mod(i,10000)
%             disp(i);
%         end
%           pn=size(saccade(i).x,1); % the number of points
%         for j=1:min(10,pn-10)
%             error(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(j))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(j))^2);
%             displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
%             displacementP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
%         end
%     
%         for j=min(11,pn-9):pn 
%             KnownSaccade.x = saccade(i).x(1:j)'; %known points of the saccade
%             KnownSaccade.y = saccade(i).y(1:j)';
%         
%             position(i,j) = PredictScd(KnownSaccade , 10); % predict the position 10 ms later
%             error(i,j) = sqrt( (position(i,j).x(end) - saccade(i).x(min(j+10,pn)))^2 + (position(i,j).y(end) - saccade(i).y(min(j+10,pn)))^2);
%             displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
%             displacementP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
%         end
%      %   average_dif(i) = mean(difference);
%         magnitude(i) = sel_saccades(i).length;
%         % calculates the distance between both positions
%        
% end
% 
% save('ShuhangPrediction.mat', 'error', 'magnitude', 'displacement', 'displacementP');
%  disp('Saving Columns');

 
for i=1:length(sel_saccades)%size(sel_saccades,2)
 
        if ~mod(i,10000)
            disp(i);
        end
          pn=size(saccade(i).x,1); % the number of points
        for j=1:min(3,pn-3)
            error(i,j) = sqrt( (saccade(i).x(min(j+3,pn)) - saccade(i).x(j))^2 + (saccade(i).y(min(j+3,pn)) - saccade(i).y(j))^2);
            displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
            displacementP(i,j) = sqrt( (saccade(i).x(min(j+3,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+3,pn)) - saccade(i).y(1))^2);
        end
    
        for j=min(11,pn-9):pn 
            KnownSaccade.x = saccade(i).x(1:j)'; %known points of the saccade
            KnownSaccade.y = saccade(i).y(1:j)';
        
            position(i,j) = PredictScd(KnownSaccade , 3); % predict the position 10 ms later
            error(i,j) = sqrt( (position(i,j).x(end) - saccade(i).x(min(j+3,pn)))^2 + (position(i,j).y(end) - saccade(i).y(min(j+3,pn)))^2);
            displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
            displacementP(i,j) = sqrt( (saccade(i).x(min(j+3,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+3,pn)) - saccade(i).y(1))^2);
        end
     %   average_dif(i) = mean(difference);
        magnitude(i) = sel_saccades(i).length;
        % calculates the distance between both positions
       
end

save('ShuhangPrediction.mat', 'error', 'magnitude', 'displacement', 'displacementP');



cols = [];
coli = 1;

theSize = length(magnitude);
for i=1:theSize %size(sel_saccades,2)
    if ~mod(i,10000)
        disp(i);
        save('colsShuhang3.mat', 'cols');
    end
    
    index = find (~error(i,:));
    if numel(index)>0
        numberrows = index(1);
    else
        numberrows = size(error,2);
    end
        
    % first column is the error
    cols(coli:coli-1+numberrows,1)=error(i,1:numberrows);
    % then the timing sample
    cols(coli:coli-1+numberrows,2)=1:numberrows;
    % then the magnitude
    cols(coli:coli-1+numberrows,3)=magnitude(i);
    % then the percentage
    cols(coli:coli-1+numberrows,4)=displacement(i,1:numberrows)/magnitude(i);
    cols(coli:coli-1+numberrows,5)=displacementP(i,1:numberrows)/magnitude(i);
    
    coli = coli+numberrows-1;
end

save('colsShuhang3', 'cols');




disp('Calling No prediction');
% These are the error with the 10 ms before (no prediction)
%The displacement in the predicted sample
for i=1:length(sel_saccades)%size(sel_saccades,2)
 
        if ~mod(i,10000)
            disp(i);
        end
         pn=length(saccade(i).x); % the number of p
         
         for j=1:pn
             errorN(i,j) = sqrt( (saccade(i).x(min(pn,j+10)) - saccade(i).x(j))^2 + (saccade(i).y(min(pn,j+10)) - saccade(i).y(j))^2);
             displacementN(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
         end
        
        magnitudeN(i) = sel_saccades(i).length'; 
        
       % magnitudeS(i) = sel_saccades(i).length';
%         pn=size(saccade(i).x,1); % the number of points
%         for j=1:pn - 10
%             displacementP(i,j) = sqrt(  (saccade(i).x(j+10) - saccade(i).x(1))^2 + (saccade(i).y(j+10) - saccade(i).y(1))^2);
%         end
       
end

save('NoPrediction.mat', 'errorN', 'displacementN', 'magnitudeN');

colsN = [];
coli = 1;
 disp('Saving Columns');

theSize = length(magnitudeN);
for i=1:theSize %size(sel_saccades,2)
    if ~mod(i,10000)
        disp(i);
    end
    
    index = find (~errorN(i,:));
    if numel(index)>0
        numberrows = index(1);
    else
        numberrows = size(errorN,2);
    end
    
    % first column is the error
    colsN(coli:coli-1+numberrows,1)=errorN(i,1:numberrows);    
    % then the timing sample
    colsN(coli:coli-1+numberrows,2)=1:numberrows;
    % then the magnitude
    colsN(coli:coli-1+numberrows,3)=magnitudeN(i);
    % then the percentage
    colsN(coli:coli-1+numberrows,4)=displacementN(i,1:numberrows)/magnitudeN(i);
   
    
    coli = coli+numberrows-1;
end

save('colsN', 'colsN');

            

disp('Calling Peng');
% Calling Peng algorithm
for i=1:length(sel_saccades)%size(sel_saccades,2)
 
        if ~mod(i,10000)
            disp(i);            
            save('Peng.mat', 'errorPeng', 'magnitudePeng', 'displacementPeng');
        end
          pn=size(saccade(i).x,1); % the number of points
        for j=1:min(10,pn-10)       
            errorPeng(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(j))^2 + (saccade(i).y(min(pn,j+10)) - saccade(i).y(j))^2);
            displacementPeng(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
            displacementPengP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
        end
      
        for j=min(11,pn-9):pn 
            KnownSaccade.x = saccade(i).x(1:j)'; %known points of the saccade
            KnownSaccade.y = saccade(i).y(1:j)';
        
            position(i,j) = PredictScd_Peng(KnownSaccade , 10); % predict the position 10 ms later
            errorPeng(i,j) = sqrt( (position(i,j).x(end) - saccade(i).x(min(pn,j+10)))^2 + (position(i,j).y(end) - saccade(i).y(min(pn,j+10)))^2);          
            displacementPengP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
            displacementPeng(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
            %dPeng(i,j) = sqrt( (saccade(i).x(j+10) - saccade(i).x(1))^2 + (saccade(i).y(j+10) - saccade(i).y(1))^2);
        end
    %    average_difPeng(i) = mean(difference);
        magnitudePeng(i) = sel_saccades(i).length;
        % calculates the distance between both positions
       
end
save('PengPredictionAgain.mat', 'errorPeng', 'magnitudePeng', 'displacementPeng', 'displacementPengP');



colsPeng = [];
coli = 1;
 

 disp('Saving Columns');

 
theSize = length(magnitudePeng);
for i=1:theSize %size(sel_saccades,2)
        if ~mod(i,10000)
            disp(i);
              save('colsPeng.mat', 'colsPeng');
        end    
        
         index = find (~errorPeng(i,:));
         if numel(index)>0
        numberrows = index(1);
         else
            numberrows = size(errorPeng,2);
         end
        
        
   % first column is the error
   colsPeng(coli:coli-1+numberrows,1)=errorPeng(i,1:numberrows);        
   % then the timing sample
   colsPeng(coli:coli-1+numberrows,2)=1:numberrows;
   % then the magnitude
   colsPeng(coli:coli-1+numberrows,3)=magnitudePeng(i);
   % then the percentage
   colsPeng(coli:coli-1+numberrows,4)=displacementPeng(i,1:numberrows)/magnitudePeng(i);
  % colsPeng(coli:coli-1+numberrows,5)=displacementPengP(i,1:numberrows)/magnitudePeng(i);
     
   coli = coli+numberrows-1;
end

save('colsPeng', 'colsPeng');


%This will display in console the percentage of errors larger than 2 degrees
100*sum(averag > 2)/length(sel_saccades)

figure;plot([0:0.01:5],histc(difference,[0:0.01:5]), 'LineWidth', 2 , 'Color', [0 0 0]);
xlabel('Error (degrees)', 'FontSize', 18, 'FontName', 'Arial');
ylabel('N', 'FontSize', 18, 'FontName', 'Arial');
set(gca, 'FontSize', 16', 'FontName', 'Arial');
box off





%%%% NEW PENG



disp('Calling Shuhang');

% Calling Shuhang algorithm

% for i=1:length(sel_saccades)%size(sel_saccades,2)
%  
%         if ~mod(i,10000)
%             disp(i);
%         end
%           pn=size(saccade(i).x,1); % the number of points
%         for j=1:min(10,pn-10)
%             error(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(j))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(j))^2);
%             displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
%             displacementP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
%         end
%     
%         for j=min(11,pn-9):pn 
%             KnownSaccade.x = saccade(i).x(1:j)'; %known points of the saccade
%             KnownSaccade.y = saccade(i).y(1:j)';
%         
%             position(i,j) = PredictScd(KnownSaccade , 10); % predict the position 10 ms later
%             error(i,j) = sqrt( (position(i,j).x(end) - saccade(i).x(min(j+10,pn)))^2 + (position(i,j).y(end) - saccade(i).y(min(j+10,pn)))^2);
%             displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
%             displacementP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
%         end
%      %   average_dif(i) = mean(difference);
%         magnitude(i) = sel_saccades(i).length;
%         % calculates the distance between both positions
%        
% end
% 
% save('ShuhangPrediction.mat', 'error', 'magnitude', 'displacement', 'displacementP');
%  disp('Saving Columns');

 
for i=1:length(sel_saccades)%size(sel_saccades,2)
 
        if ~mod(i,10000)
            disp(i);
        end
          pn=size(saccade(i).x,1); % the number of points
        for j=1:min(10,pn-10)
            error(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(j))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(j))^2);
            displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
            displacementP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
        end
    
        for j=min(11,pn-9):pn 
            KnownSaccade.x = saccade(i).x(1:j)'; %known points of the saccade
            KnownSaccade.y = saccade(i).y(1:j)';
        
            position(i,j) = PredictScd_Peng(KnownSaccade , 10); % predict the position 10 ms later
            error(i,j) = sqrt( (position(i,j).x(end) - saccade(i).x(min(j+10,pn)))^2 + (position(i,j).y(end) - saccade(i).y(min(j+10,pn)))^2);
            displacement(i,j) = sqrt( (saccade(i).x(j) - saccade(i).x(1))^2 + (saccade(i).y(j) - saccade(i).y(1))^2);
            displacementP(i,j) = sqrt( (saccade(i).x(min(j+10,pn)) - saccade(i).x(1))^2 + (saccade(i).y(min(j+10,pn)) - saccade(i).y(1))^2);
        end
     %   average_dif(i) = mean(difference);
        magnitude(i) = sel_saccades(i).length;
        % calculates the distance between both positions
       
end

save('PengPrediction.mat', 'error', 'magnitude', 'displacement', 'displacementP');



cols = [];
coli = 1;

theSize = length(magnitude);
for i=1:theSize %size(sel_saccades,2)
    if ~mod(i,10000)
        disp(i);
        save('colsPeng10.mat', 'cols');
    end
    
    index = find (~error(i,:));
    if numel(index)>0
        numberrows = index(1);
    else
        numberrows = size(error,2);
    end
        
    % first column is the error
    cols(coli:coli-1+numberrows,1)=error(i,1:numberrows);
    % then the timing sample
    cols(coli:coli-1+numberrows,2)=1:numberrows;
    % then the magnitude
    cols(coli:coli-1+numberrows,3)=magnitude(i);
    % then the percentage
    cols(coli:coli-1+numberrows,4)=displacement(i,1:numberrows)/magnitude(i);
    cols(coli:coli-1+numberrows,5)=displacementP(i,1:numberrows)/magnitude(i);
    
    coli = coli+numberrows-1;
end

save('colsPeng10', 'cols');





