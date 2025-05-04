%Name – Ahaan Tagare
%Student ID - 33865799

                                        %Coursework 1 Part-B

% Initial weights for the first input
% w1: -0.1, w2: -0.3, w3: -0.2, w4: 0.2, w5: 0.1, w6: -0.1, w7: 0.2, w8: -0.3, w9: 0.3 
% Weights 10 and 11 are taken as 0 to connect x1 to sigmoid 3 and and x2 to sigmoid 1 to create a fully connected neural network. 

% Initial weights
weights = [-0.1, -0.3, -0.2, 0.2, 0.1, -0.1, 0.2, -0.3, 0.3, 0, 0];

% Learning rate
eta = 0.2;

% Training data
training_data = [
    0, 1, 1;  % Input 1: x1 = 0, x2 = 1, t = 1
    1, 0, 1   % Input 2: x1 = 1, x2 = 0, t = 1
];

% Define sigmoid activation function
sigmoid = @(x) 1 / (1 + exp(-x));

% Indices for weights
hidden_indices = [1, 11; 3, 4; 10, 6];  % Weights for hidden neurons
output_weights = [2, 5, 9, 8, 7];       % Weights for output connections

% Loop through training examples
for example = 1:size(training_data, 1)
    % Extract inputs and target from the current training example
    inputs = training_data(example, 1:2);
    target = training_data(example, 3);
    
    % Forward pass: Calculate outputs of hidden layer neurons
    y = zeros(1, 3);  % Initialize outputs of hidden neurons
    for j = 1:3
        net_hidden = 0;  % Net input for hidden neuron j
        for i = 1:2
            net_hidden = net_hidden + weights(hidden_indices(j, i)) * inputs(i);
        end
        y(j) = sigmoid(net_hidden);  % Apply activation function
    end
    
    % Forward pass: Calculate output
    output = 0;  % Initialize output
    % Direct input-to-output connections
    for i = 1:2
        output = output + weights(output_weights(i)) * inputs(i);
    end
    % Hidden neuron-to-output connections
    for j = 1:3
        output = output + weights(output_weights(2 + j)) * y(j);
    end
    
    % Error calculation
    error = target - output;
    
    % Backward pass: Calculate betas
    beta_output = error;  % Beta for output node
    beta_hidden = zeros(1, 3);  % Betas for hidden neurons
    delta_weights = zeros(size(weights));  % Initialize weight updates
    
    % Update weights from inputs to hidden neurons
    for j = 1:3
        % Calculate beta for hidden neuron j
        beta_hidden(j) = y(j) * (1 - y(j)) * beta_output * weights(output_weights(2 + j));
        for i = 1:2
            % Update weights for input-to-hidden connections
            delta_weights(hidden_indices(j, i)) = eta * beta_hidden(j) * inputs(i);
        end
    end
    
    % Update weights for direct input-to-output connections
    for i = 1:2
        delta_weights(output_weights(i)) = eta * beta_output * inputs(i);
    end
    
    % Update weights for hidden neuron-to-output connections
    for j = 1:3
        delta_weights(output_weights(2 + j)) = eta * beta_output * y(j);
    end
    
    % Apply weight updates
    weights = weights + delta_weights;
    
    % Display results for the current training example
    fprintf('Results for Training Example %d:\n', example);
    fprintf('Inputs: x1 = %.1f, x2 = %.1f\n', inputs(1), inputs(2));
    fprintf('Target Output: %.1f\n', target);
    fprintf('Actual Output: %.4f\n', output);
    fprintf('Error: %.4f\n', error);
    
    for j = 1:3
        fprintf('Hidden Neuron %d Output (y%d): %.4f\n', j, j, y(j));
    end
    
    fprintf('Beta for Output Node: %.4f\n', beta_output);
    for j = 1:3
        fprintf('Beta for Hidden Neuron %d: %.4f\n', j, beta_hidden(j));
    end
    
    fprintf('\nWeight Updates for Training Example %d:\n', example);
    disp(delta_weights);
    
    fprintf('Updated Weights After Training Example %d:\n', example);
    for i = 1:length(weights)
        fprintf('w%d = %.4f\n', i, weights(i));
    end
    fprintf('\n');
end



% Update the new weights (Same as before)
weights = [-0.1, -0.3, -0.2, 0.1878, 0.2640, -0.0918, 0.2779, -0.2098, 0.3820, 0, 0.0123];

% Define inputs and target for the second example
inputs = [1, 0];
target = 1;

% Initialize variables for the hidden layer and output
y = zeros(1, 3);  % Outputs of hidden neurons
net_hidden = zeros(1, 3); 

% Forward pass: 
for j = 1:3
    net_hidden(j) = 0;  % Reset net input for each hidden unit
    for i = 1:2
       
        net_hidden(j) = net_hidden(j) + weights(hidden_indices(j, i)) * inputs(i);
    end
    y(j) = sigmoid(net_hidden(j));
end

% Determine the network's output -  total of hidden layer's contribution
output = weights(output_weights(1)) * inputs(1) + weights(output_weights(2)) * inputs(2) + ...
         sum(weights(output_weights(3:end)) .* y);

% Error calculation
error = target - output;

% Backward pass: betas for each layer
beta_output = error;  % Beta output layer

% Initialize delta_weights to store weight updates
delta_weights = zeros(size(weights));

% Calculate betas for the hidden neurons and weight updates
for j = 1:3
    beta(j) = y(j) * (1 - y(j)) * beta_output * weights(output_weights(2 + j));
   
    for i = 1:2
      
        net_hidden(j) = 0;
        net_hidden(j) = net_hidden(j) + weights(hidden_indices(j, i)) * inputs(i);
        
        
        delta_weights(hidden_indices(j, i)) = eta * beta(j) * inputs(i);
    end
end

% Update weights from input to output (first two weights)
delta_weights(output_weights(1:2)) = eta * beta_output * inputs;

% Update weights from hidden to output (remaining weights)
delta_weights(output_weights(3:end)) = eta * beta_output * y;

% Apply weight updates
weights = weights + delta_weights;


