% EdX CS1156x Learning from Data Final Exam, Problem 8
% Author: David Eisner (deisner@gmail.com)

function E_out_table = p8final()

% data columns: digit, symmetry, intensity
D_train = importdata('features.train');
D_test  = importdata('features.test');

% Din = [din(:,1) din(:,2)];
N_train = size(D_train,1);
N_test  = size(D_test,1);

lambda = 1;
verses = 0:4;

E_out_table = zeros(numel(verses),2);

i = 1;
for digit = verses
    
    y_train = ones(N_train,1);
    y_train(D_train(:,1) ~= digit) = -1;
    
    Z_train = phi( D_train(:,2:3));
    w = linear_reg_decay_w(Z_train, y_train, lambda);
    
    y_test = ones(N_test, 1);
    y_test( D_test(:,1) ~= digit) = -1;
    Z_test = phi( D_test(:,2:3));
    
    y_pred = sign( Z_test*w);
    E_out = sum(y_test ~= y_pred)/numel(y_test);
    E_out_table(i,:) = [digit E_out];
    i=i+1;
end

disp(E_out_table);
end

function w_reg = linear_reg_decay_w(Z, y, lambda)
N = size(Z,2);
w_reg = ((Z'*Z + lambda*eye(N))\Z')*y;
end

% D = [x1_1, x2_1; x1_2 x2_2; ... ]
% returns each row of D tranformed to 
% (1, x1, x2, x1*x2, x1^2, x2^2)
function Z = phi(D)
    N = size(D,1);
    X0 = ones(N,1);
    X1 = D(:,1);
    X2 = D(:,2);
    Z = [X0, X1, X2,  X1.*X2, X1.^2, X2.^2];
end


