
load_MNIST_data;
addpath layers;

batch_size = 100;
numIters = 10;
num_batches = 10;
epho_num = 1;
train_loss = zeros(numIters, epho_num);
test_loss_list = zeros(num_batches, epho_num);
accuracy_list = zeros(num_batches, epho_num);
acc_val = zeros(epho_num, 1);

for epho = 1 : epho_num
    num_node = 8 - (epho - 1) * 2;
    l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',num_node))
	    init_layer('pool',struct('filter_size',4,'stride',2))
	    init_layer('relu',[])
	    init_layer('flatten',struct('num_dims',4))
	    init_layer('linear',struct('num_in',121* num_node,'num_out',10))
	    init_layer('softmax',[])];

    model = init_model(l,[28 28 1],10,true);
 
    params=struct('learning_rate',.15,'weight_decay',0.0002,'batch_size',batch_size);
    
    [model, loss] = train(model,train_data,train_label,params,numIters);
    train_loss(:, epho) = loss;
    
    %show result
    params1=struct('batch_size',batch_size);
    
    test_acc=zeros(num_batches,1);
    test_loss=zeros(num_batches,1);
    
    % measure test time
    tic
    
    parfor i=1:num_batches
        % Select non-random batches
        input_batch=test_data(:,:,:,(i-1)*batch_size+1:i*batch_size);
        label_batch=test_label((i-1)*batch_size+1:i*batch_size,:);
        [final_layer_output,~] = inference(model,input_batch);
        inferred_label=zeros(size(label_batch));
        for j=1:batch_size
            [~,inferred_label(j)]=max(final_layer_output(:,j));
        end
        curr_test_acc=inferred_label==label_batch;
        test_acc(i)=sum(curr_test_acc)/batch_size;
        [test_loss(i), ~] = loss_crossentropy(final_layer_output, label_batch, [], false);
    end
    accuracy = mean(test_acc);
    curr_loss = mean(test_loss);
    accuracy_list(:, epho) = test_acc;
    test_loss_list(:, epho) = test_loss; 
    acc_val(epho) = accuracy;    
    toc
    
end    
  
plot(train_loss(:, 1)); 
hold on;
plot(train_loss(:, 2)); 
hold on;
plot(train_loss(:, 3)); 
hold on;
plot(train_loss(:, 4)); legend('num node : 8','num node : 6','num node : 4','num node : 2'); 
hold on;



plot(test_loss_list(:, 1)); 
hold on;
plot(test_loss_list(:, 2)); 
hold on;
plot(test_loss_list(:, 3)); 
hold on;
plot(test_loss_list(:, 4)); legend('num node : 8','num node : 6','num node : 4','num node : 2'); 
hold on;