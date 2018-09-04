function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just looping in reverse.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

% TODO: Determine the gradient at each layer with weights to be updated
for i = num_layers : -1 : 1
    if i == 1
        input_img = input;
    else
        input_img = activations{i-1};
    end

    [~, dv_input , grad{i}] = model.layers(i).fwd_fn(input_img, model.layers(i).params, model.layers(i).hyper_params, true, dv_output);
    dv_output = dv_input;
end


end
