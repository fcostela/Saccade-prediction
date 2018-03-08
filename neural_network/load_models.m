%Script to load the different neural networks models depending of the
%available data size

file_list=dir(['./models/m*.*']);
nn_model=struct('Iw',{},'b',{},'LW',{},'norm_param_in',{},'norm_param_out',{},'preinfo',{});
for ifile=1:length(file_list)
    file_name=file_list(ifile).name;
    load(['./models/',file_name]);   
        
    itrain=str2num(file_name(end-5:end-4));

    nn_model(itrain).Iw=Iw;
    nn_model(itrain).b=b;
    nn_model(itrain).LW=LW;
    nn_model(itrain).norm_param_in=norm_param_in;
    nn_model(itrain).norm_param_out=norm_param_out; 
    nn_model(itrain).preinfo=preinfo;  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  How to predict  %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

saccade_data=saccadeIndex(1).saccadeNoFilter(16-preinfo:end,:); %Remember to add data before the saccade. The number of data points is equal to preinfo value 
itrain=size(saccade_data,1)-preinfo;
if itrain>40
    itrain=40;
end

input_data(:,1)=saccade_data(2:itrain+preinfo,1)-saccade_data(1,1); %The data is normalized according to the first sample. 
input_data(:,2)=saccade_data(2:itrain+preinfo,2)-saccade_data(1,2); %The first data point will be 0. We remove it

tic
predicted_location=predict_landing([input_data(:,1); input_data(:,2)],nn_model(itrain)); %x and y coordinates are concatenated
toc


