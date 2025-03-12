% Load CIFAR-10 dataset
load('data_batch_1.mat'); % Assume cifar10.mat contains 'data' and 'labels'

% Preprocess CIFAR-10 images
X_train = double(reshape(data, [size(data, 1), 32, 32, 3])) / 255;

% Create noisy version of the images for training
X_train_noisy = X_train + 0.1 * randn(size(X_train));

% Define the autoencoder architectures
inputSize = 32 * 32 * 3;
hiddenSize = 400;

% Train DAE
tic;
dae = trainAutoencoder(reshape(X_train_noisy, [], inputSize)', ...
    reshape(X_train, [], inputSize)', 'MaxEpochs', 50);
daeTrainingTime = toc;

% Train nlDAE (assuming trainNlDAE is your custom training function)
tic;
nldae = trainNlDAE(reshape(X_train_noisy, [], inputSize)', ...
    reshape(X_train, [], inputSize)', 'MaxEpochs', 50);
nldaeTrainingTime = toc;

% Evaluate models (using a test set)
X_test = double(reshape(data, [size(data, 1), 32, 32, 3])) / 255;
X_test_noisy = X_test + 0.1 * randn(size(X_test));

% Reconstruct images
dae_reconstructed = predict(dae, reshape(X_test_noisy, [], inputSize)');
nldae_reconstructed = predict(nldae, reshape(X_test_noisy, [], inputSize)');

% Calculate denoising accuracy (e.g., MSE)
daeMSE = mean(mean((reshape(X_test, [], inputSize)' - dae_reconstructed).^2));
nldaeMSE = mean(mean((reshape(X_test, [], inputSize)' - nldae_reconstructed).^2));

% Display results
disp(['DAE Training Time: ' num2str(daeTrainingTime)]);
disp(['nlDAE Training Time: ' num2str(nldaeTrainingTime)]);
disp(['DAE MSE: ' num2str(daeMSE)]);
disp(['nlDAE MSE: ' num2str(nldaeMSE)]);
