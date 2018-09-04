function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE
for i = 1 : num_layers
    
    [activations{i}, ~ , ~] = model.layers(i).fwd_fn(input, model.layers(i).params,model.layers(i).hyper_params, false, []);
    
    input=activations{i};
end

output = activations{end};
