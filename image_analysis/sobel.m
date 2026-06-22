% Edge Detection Visualization with Sobel Operators

% Original image
img = [0 0 1 0 0;
      0 0 1 0 0;
      0 0 1 0 0;
      0 0 1 0 0;
      0 0 1 0 0];
% Sobel operators
fx = [-1 0 1; -2 0 2; -1 0 1];  % Horizontal gradient
fy = [-1 -2 -1; 0 0 0; 1 2 1];  % Vertical gradient
% Perform convolution
w_x = conv2(img, fx, 'same');
w_y = conv2(img, fy, 'same');
% Combine gradients (magnitude of the gradient)
edge_magnitude = sqrt(w_x.^2 + w_y.^2);
% Visualize results
figure;
% Original image
subplot(2,2,1);
imagesc(img);
title('Original Image');
colorbar;
% X-direction gradient
subplot(2,2,2);
imagesc(w_x);
title('X-Direction Gradient');
colorbar;
% Y-direction gradient
subplot(2,2,3);
imagesc(w_y);
title('Y-Direction Gradient');
colorbar;
% Edge Magnitude
subplot(2,2,4);
imagesc(edge_magnitude);
title('Edge Magnitude');
colorbar;
% Adjust figure
colormap('jet');
sgtitle('Edge Detection Visualization');
