% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

[num_nodes, batch_size] = size(input);
% TODO: CALCULATE LOSS


loss = 0;
for i = 1 : batch_size
    loss = loss - log(input(labels(i), i));    
end
loss = loss / batch_size;


dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    for i = 1 : batch_size
        dv_input(labels(i), i) = -1 / input(labels(i), i);
    end
end


end