% EdX CS1156x Learning from Data Final Exam, Problem 7
% Author: David Eisner (deisner@gmail.com)

function E_in_table = p7final()

% data columns: digit, symmetry, intensity
din = importdata('features.train');

% Din = [din(:,1) din(:,2)];
N = size(din,1);

lambda = 1;
verses = 5:9;

E_in_table = zeros(numel(verses),2);

i = 1;
for digit = verses
    
    y = ones(N,1);
    y(din(:,1) ~= digit) = -1;
    
    Z = [ones(N,1) din(:,2:3)];
    w = linear_reg_decay_w(Z, y, lambda);
    
    y_in = sign( Z*w);
    E_in = sum(y ~= y_in)/numel(y);
    E_in_table(i,:) = [digit E_in];
    i=i+1;
end

disp(E_in_table);
end

function w_reg = linear_reg_decay_w(Z, y, lambda)
N = size(Z,2);
w_reg = ((Z'*Z + lambda*eye(N))\Z')*y;
end


