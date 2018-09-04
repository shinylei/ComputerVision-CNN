% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[~,~,num_channels,batch_size] = size(input);
[~,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);
% TODO: FORWARD CODE
for batch = 1 : batch_size
    for filter = 1 : num_filters
        for chan = 1 : num_channels
           output(:, :, filter, batch) = output(:, :, filter, batch) + conv2(input(:, :, chan, batch), params.W(:, :, chan, filter), 'valid');
        end
        output(:, :, filter, batch) = output(:, :, filter, batch) + params.b(filter);
    end
end


dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    
   for chan = 1 : filter_depth 
        for batch = 1 : batch_size
            for filter = 1 : num_filters
                dv_input(:,:,chan,batch) = dv_input(:,:,chan,batch) + conv2(dv_output(:, :, filter, batch), rot90(params.W(:, :, chan, filter), 2), 'full');
            end
        end
   end
    
    for chan = 1 : num_channels
        for filter = 1 : num_filters
            for batch = 1 : batch_size
                grad.W(:, :, chan, filter) = grad.W(:, :, chan, filter) + conv2(rot90(input(:, :, chan, batch), 2), dv_output(:, :, filter, batch), 'valid') / batch_size;
            end
        end
    end    
    

    for batch = 1 : size(dv_output,3)
        grad.b(batch) = sum(sum(sum(dv_output(:,:,batch,:)))) / batch_size;
    end
    
end
