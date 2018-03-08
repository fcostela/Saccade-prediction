% clear all
rng('shuffle'); %Initialization of the random seed
% load('saccadeTV1Degree_plus16msbefore.mat')

preinfo=2; % Start point two samples before the beggining of the saccade
cont=1;
saccade_matrix_train=[]; %Cell array conteining all the saccades
for ii=1:1:length(saccadeIndex)
    
    x0=saccadeIndex(ii).saccadeNoFilter(16,1);
    y0=saccadeIndex(ii).saccadeNoFilter(16,2);
    x1=saccadeIndex(ii).saccadeNoFilter(end,1);
    y1=saccadeIndex(ii).saccadeNoFilter(end,2);
    ls=sqrt(((x0-x1).^2)+((y0-y1).^2)); %remove saccades greater than 40 degrees because they are probably errors;
    if ls<40        
        saccade_matrix_train{cont,1}=(saccadeIndex(ii).saccadeNoFilter(16-preinfo:end,1)-saccadeIndex(ii).saccadeNoFilter(16-preinfo,1));
        saccade_matrix_train{cont,2}=(saccadeIndex(ii).saccadeNoFilter(16-preinfo:end,2)-saccadeIndex(ii).saccadeNoFilter(16-preinfo,2));
         
        cont=cont+1;
    end
end



for itrain=16:1:50 %itrain is the number of samples used for the prediction. There are different strategies to avoid to use one model for each length (e.g. zero padding,...). However, the best results I have obtained was training one network for each length 
        
    input_matrix=[]; % Input of the Netword
    target_matrix=[]; %Desirable output of the network
    
    for ii=1:1:length(saccade_matrix_train)        
        if length(saccade_matrix_train{ii,1})>itrain+preinfo+2  %we will use only    samples with length greater than itrain+preinfo +2 (the +2 is only to add certain temporaly distance)       
                       
            input_matrix(:,end+1)=[saccade_matrix_train{ii,1}(2:itrain+preinfo); saccade_matrix_train{ii,2}(2:itrain+preinfo)]; %First element removed (always equal to 0)
            target_matrix(:,end+1)=[saccade_matrix_train{ii,1}(end); saccade_matrix_train{ii,2}(end)];
            
        end        
    end
    
    net = feedforwardnet([32 32]); %here we define our network, a feedforward net with 2 layers of 32 units each one
    %         view(net)
    if size(input_matrix,2)<90000 %it is not necessary to train with all samples
        step=1;
    else
        step=2;
    end
    validation_samples=3000; %number of samples used for the validation of the netwrok
    net.input.processFcns={'removeconstantrows','mapstd'}; %parameteers of the network training
    [net,tr] = train(net,input_matrix(:,1:step:end-validation_samples),target_matrix(:,1:step:end-validation_samples)); %training. Note that step help to reduce the number of samples in case we have more than 90000.
    
    %The next parameteers define our trained network 
    Iw=net.IW; %input weight
    b=net.b; %bias
    LW=net.LW; %Layer weights
    norm_param_in=net.input.processSettings{1}; %input normalization parameteers
    norm_param_out=net.output.processSettings{1}; %output normalization parameteers
    
    %Now we save out models into .mat files
    fname=['model2/model_32x32_itrain_',num2str(itrain),'.mat'];
    save(fname,'Iw','b','LW','norm_param_in','norm_param_out','preinfo') %You dont need the neural network toolbox to use the parameteers
    fname=['model2/net_32x32_itrain_',num2str(itrain),'.mat']; 
    save(fname,'net','input_matrix'); %You need the toolbox tu execute the network net. Do not need to copy this files into the demo
%     
    
    %validation of the network
    testI=input_matrix(:,end-validation_samples:end);
    testT=target_matrix(:,end-validation_samples:end);
    testY = net(testI);
    avg_error=mean(sqrt(((testT(1,:)-testY(1,:)).^2)+((testT(2,:)-testY(2,:)).^2)));
    fprintf(['Mean Error:',num2str(avg_error),'\n'])
    
    
end

