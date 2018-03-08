function [out] = predict_landing(in,nn_model)

num_layers=size(nn_model.LW,1)-1;

input_n=((in-nn_model.norm_param_in.xmean).*(nn_model.norm_param_in.ystd./nn_model.norm_param_in.xstd))+nn_model.norm_param_in.ymean;     
Ai=2./(1+exp(-2.*((nn_model.Iw{1}*input_n)+nn_model.b{1})))-1;
for ii=2:num_layers   
    Ai=2./(1+exp(-2.*((nn_model.LW{ii,ii-1}*Ai)+nn_model.b{ii})))-1;    
end
An=(nn_model.LW{num_layers+1,num_layers}*Ai)+nn_model.b{num_layers+1};
out=(((An+1)/2).*((nn_model.norm_param_out.xmax-nn_model.norm_param_out.xmin)))+nn_model.norm_param_out.xmin;