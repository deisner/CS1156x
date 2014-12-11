% EdX CS1156x Learning from Data Final Exam, Problem 10
% Author: David Eisner (deisner@gmail.com)

function E_table = p10final()

% data columns: digit, symmetry, intensity
D_train = importdata('features.train');
D_test  = importdata('features.test');

d1 = 1;
d2 = 5;

% Filter: Keep only rows for digits d1 and d2 (for a d1 vs d2 classifier)
D_train = D_train( (D_train(:,1) == d1) | (D_train(:,1) == d2),:);
D_test  = D_test(  (D_test(:,1)  == d1) | (D_test(:,1)  == d2),:);

% Din = [din(:,1) din(:,2)];
N_train = size(D_train,1);
N_test  = size(D_test,1);

lambdas = [0.01 1];

% summarize results in a table. Each row:
% lambda, E_in, E_out
E_table = zeros(numel(lambdas),3);
y_train = ones(N_train,1);
y_train(D_train(:,1) ~= d1) = -1;

y_test = ones(N_test, 1);
y_test( D_test(:,1) ~= d1) = -1;

Ztrans_train = phi( D_train(:,2:3));
Ztrans_test  = phi( D_test(:,2:3));

i = 1;
for lambda = lambdas;

    w = linear_reg_decay_w(Ztrans_train, y_train, lambda);
    
    % E_in
    y_pred = sign(Ztrans_train*w);
    E_in = sum(y_pred ~= y_train)/numel(y_train);
    
    % E_out
    y_pred = sign(Ztrans_test*w);
    E_out = sum(y_pred ~= y_test)/numel(y_test);
  
    E_table(i,:) = [lambda E_in E_out];
    i=i+1;
end

disp(E_table);
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


