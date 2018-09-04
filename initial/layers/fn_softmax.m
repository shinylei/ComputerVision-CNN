% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
% TODO: FORWARD CODE
expo = exp(input);
output = expo./repmat(sum(expo,1), [num_classes,1]);

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
    dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    for i = 1 : batch_size
       y = output(:, i);
       dvx = diag(y) - y * y';
       dv_input(:, i) = dvx * dv_output(:, i);
    end
    
end
